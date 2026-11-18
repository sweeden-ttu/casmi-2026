# [Colab Cell 7] Stage 2: Context-Aware Generative Training (Cross-Fusion + Decoder)
@torch.no_grad()
def build_candidate_library_index(
    model: GLMRPipeline,
    candidate_smiles: List[str],
    tokenizer: SMILESTokenizer,
    batch_size: int = 256,
    device: torch.device = CFG.DEVICE,
) -> Tuple[List[str], np.ndarray, torch.Tensor]:
    """
    Encodes candidate database (train structures + PubChem/COCONUT) into normalized
    ChemFormer embeddings E^m and records exact neutral monoisotopic masses for ppm filtering.
    """
    from rdkit.Chem import Descriptors

    model.eval()
    valid_smiles, exact_masses, emb_batches = [], [], []

    for i in range(0, len(candidate_smiles), batch_size):
        chunk = candidate_smiles[i : i + batch_size]
        toks = []
        for smi in chunk:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            valid_smiles.append(smi)
            exact_masses.append(Descriptors.ExactMolWt(mol))
            toks.append(tokenizer.encode(smi, add_cls=True))
        if not toks:
            continue
        tok_tensor = torch.stack(toks).to(device)
        _, e_m = model.mol_encoder(tok_tensor)
        emb_batches.append(e_m.cpu())

    library_embs = torch.cat(emb_batches, dim=0) if emb_batches else torch.empty(0, CFG.D_MODEL)
    return valid_smiles, np.asarray(exact_masses, dtype=np.float64), library_embs


def in_mass_window(masses: np.ndarray, neutral_mass: float, ppm_tol: float = CFG.MASS_TOLERANCE_PPM) -> np.ndarray:
    """Boolean mask of candidate exact masses within the ppm window (with the absolute floor)."""
    tol_da = max(neutral_mass * ppm_tol * 1e-6, CFG.MIN_TOLERANCE_DA)
    return np.abs(np.asarray(masses, dtype=np.float64) - neutral_mass) <= tol_da


@torch.no_grad()
def retrieve_top_k_candidates(
    e_s_batch: torch.Tensor,
    precursor_mzs: torch.Tensor,
    adducts: List[str],
    lib_masses: np.ndarray,
    lib_embs: torch.Tensor,
    top_k: int = CFG.TOP_K_CANDIDATES,
    ppm_tol: float = CFG.MASS_TOLERANCE_PPM,
    ionization_modes: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Pre-retrieves top-K candidate embeddings H^m_K per molecule using mass-constrained
    cosine similarity R(E^s, E^m_i). Falls back to global top-K if fewer than K mass hits exist.
    """
    device = e_s_batch.device
    lib_embs_dev = lib_embs.to(device)
    sim_matrix = torch.matmul(e_s_batch, lib_embs_dev.T)  # (B, N_lib)

    selected_embs = []
    selected_indices = []

    for b in range(e_s_batch.size(0)):
        prec_mz = float(precursor_mzs[b].item())
        mode = ionization_modes[b] if ionization_modes is not None else "positive"
        # Handles formate/acetate spelling, [M]+, and [2M+X] multimers (the old shift table did not).
        est_neutral_mass = neutral_mass_from_precursor(prec_mz, adducts[b], mode)

        mass_tol_da = max(est_neutral_mass * (ppm_tol * 1e-6), CFG.MIN_TOLERANCE_DA)
        mass_mask = np.abs(lib_masses - est_neutral_mass) <= mass_tol_da
        candidate_idx = np.where(mass_mask)[0]

        scores_b = sim_matrix[b]
        if len(candidate_idx) >= top_k:
            cand_t = torch.tensor(candidate_idx, dtype=torch.long, device=device)
            sub_scores = scores_b[cand_t]
            topk_local = torch.topk(sub_scores, k=top_k).indices
            top_idx = cand_t[topk_local]
        else:
            # Fewer than K structures in the mass window (the usual case at 15 ppm): keep all of them
            # first and fill the remaining slots from global similarity. The original code discarded the
            # mass-matched candidates here and returned a purely global top-K.
            k_total = min(top_k, len(lib_masses))
            cand_t = torch.tensor(candidate_idx, dtype=torch.long, device=device)
            in_window = cand_t[torch.argsort(scores_b[cand_t], descending=True)] if len(candidate_idx) else cand_t
            masked = scores_b.clone()
            masked[in_window] = float("-inf")
            fill = torch.topk(masked, k=k_total - len(in_window)).indices
            top_idx = torch.cat([in_window, fill])

        selected_indices.append(top_idx.cpu().tolist())
        selected_embs.append(lib_embs_dev[top_idx])

    return torch.stack(selected_embs, dim=0), selected_indices


def train_stage2_epoch(
    model: GLMRPipeline,
    loader: DataLoader,
    lib_masses: np.ndarray,
    lib_embs: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    pad_id: int,
    device: torch.device = CFG.DEVICE,
    deadline: Optional["ENV.Deadline"] = None,
) -> float:
    """
    GLMR Section 3.2:
    Freezes both MolecularEncoder and SpectralEncoder; trains only SpecMolCrossFusion
    and ChemFormerMolecularDecoder via autoregressive cross-entropy loss L_gen (Eq. 8).
    """
    model.train()
    model.mol_encoder.eval()
    model.spec_encoder.eval()
    model.aggregator.eval()

    for module in (model.mol_encoder, model.spec_encoder, model.aggregator):
        for p in module.parameters():
            p.requires_grad = False

    total_loss, n_batches = 0.0, 0
    for batch in loader:
        if deadline is not None and deadline.expired():
            break
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            h_s_mol, e_s_mol, _ = model.encode_molecule_spectra(batch)
            h_m_k, _ = retrieve_top_k_candidates(
                e_s_mol,
                batch["precursor_mz"],
                batch["primary_adduct"],
                lib_masses,
                lib_embs,
                top_k=CFG.TOP_K_CANDIDATES,
                ionization_modes=batch["ionization_mode"],
            )

        # Fuse spectral sequence states with top-K retrieved molecular priors
        h_ca = model.cross_fusion(h_s_mol, h_m_k)

        # Autoregressive teacher forcing: predict y_{1..Q} from y_{0..Q-1}
        dec_tokens = batch["mol_tokens_dec"]
        inp_tokens = dec_tokens[:, :-1]
        tgt_tokens = dec_tokens[:, 1:]

        logits = model.mol_decoder(inp_tokens, h_ca)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_tokens.reshape(-1),
            ignore_index=pad_id,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.cross_fusion.parameters()) + list(model.mol_decoder.parameters()),
            1.0,
        )
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
# [Colab Cell 6] Stage 1: Dual-Path InfoNCE Contrastive Alignment with Intensity-Perturbed Negatives
def dual_path_infonce_loss(
    e_mol: torch.Tensor,
    e_spec: torch.Tensor,
    e_spec_perturbed: torch.Tensor,
    tau: float = CFG.TEMPERATURE,
) -> torch.Tensor:
    """
    GLMR Equations (3)-(5):
      - L_ms2mol: in-batch molecular negatives
      - L_mol2ms: positive spectrum vs. intensity-perturbed negative spectrum + in-batch spectra
      - L_pre = 0.5 * (L_ms2mol + L_mol2ms)
    """
    B = e_mol.size(0)
    labels = torch.arange(B, device=e_mol.device)

    # MS-to-Molecule alignment (Equation 4)
    logits_ms2mol = torch.matmul(e_spec, e_mol.T) / tau
    loss_ms2mol = F.cross_entropy(logits_ms2mol, labels)

    # Molecule-to-MS alignment with intensity-perturbed hard negatives (Equation 3)
    combined_spec = torch.cat([e_spec, e_spec_perturbed], dim=0)  # (2B, D)
    logits_mol2ms = torch.matmul(e_mol, combined_spec.T) / tau    # (B, 2B)
    loss_mol2ms = F.cross_entropy(logits_mol2ms, labels)

    return 0.5 * (loss_ms2mol + loss_mol2ms)


def load_pretrained_mol_encoder(model: "GLMRPipeline", ckpt_path: str) -> bool:
    """Loads a pre-trained ChemFormer encoder state_dict if one is provided. Returns True when loaded."""
    if not ckpt_path or not Path(ckpt_path).exists():
        ENV.log("No pre-trained ChemFormer checkpoint: the molecule encoder trains jointly in Stage 1.")
        return False
    state = torch.load(ckpt_path, map_location="cpu")
    model.mol_encoder.load_state_dict(state.get("state_dict", state))
    ENV.log(f"Loaded pre-trained ChemFormer encoder from {ckpt_path}; it stays frozen in Stage 1.")
    return True


def train_stage1_epoch(
    model: GLMRPipeline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device = CFG.DEVICE,
    freeze_mol_encoder: bool = False,
    deadline: Optional["ENV.Deadline"] = None,
) -> float:
    """
    Updates SpectralEncoder + MultiSpectrumAggregator against the molecule encoder.

    GLMR freezes a *pre-trained* ChemFormer here. The original cell froze a randomly
    initialized encoder (no checkpoint was ever loaded), so spectra were aligned to random
    molecule embeddings. Freeze only when load_pretrained_mol_encoder() succeeded; otherwise
    both towers train jointly, CLIP-style.
    """
    model.train()
    if freeze_mol_encoder:
        model.mol_encoder.eval()
    for p in model.mol_encoder.parameters():
        p.requires_grad = not freeze_mol_encoder

    total_loss, n_batches = 0.0, 0
    for batch in loader:
        if deadline is not None and deadline.expired():
            break
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        if freeze_mol_encoder:
            with torch.no_grad():
                _, e_m = model.mol_encoder(batch["mol_tokens_cls"])
        else:
            _, e_m = model.mol_encoder(batch["mol_tokens_cls"])

        # Positive multi-spectrum pass
        _, e_s, _ = model.encode_molecule_spectra(batch)

        # Construct intensity-perturbed spectral negative (GLMR Section 3.1)
        perturbed_batch = dict(batch)
        noise = torch.empty_like(batch["ints"]).uniform_(W("nb1.loss.perturb_low"), W("nb1.loss.perturb_high"))
        perturbed_ints = torch.clamp(batch["ints"] * noise, 0.0, 2.0)
        perturbed_ints[:, :, 0] = 2.0  # Keep precursor anchor intact
        perturbed_batch["ints"] = perturbed_ints
        _, e_s_neg, _ = model.encode_molecule_spectra(perturbed_batch)

        loss = dual_path_infonce_loss(e_m, e_s, e_s_neg, tau=CFG.TEMPERATURE)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
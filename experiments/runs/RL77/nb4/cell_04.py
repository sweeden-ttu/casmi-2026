# [Colab Formula Filter Cell 4] Batch Integration Hook & End-to-End Verification
INV_ADDUCT_VOCAB = {idx: name for name, idx in ADDUCT_VOCAB.items()}


def molecule_formula_inputs(
    batch: Dict[str, torch.Tensor],
    b: int,
    subformula_filter: FastMISTSubformulaFilter,
) -> Dict[str, object]:
    """Consensus neutral mass and pooled peak arrays for molecule b of a MultiSpectrumMoleculeDataset batch."""
    mzs_np = batch["mzs"][b].detach().cpu().numpy()            # (S, P)
    ints_np = batch["ints"][b].detach().cpu().numpy()          # (S, P)  sqrt-scaled intensities
    nls_np = batch["nls"][b].detach().cpu().numpy()            # (S, P)
    bpi_np = 10.0 ** batch["bpi"][b].detach().cpu().numpy()    # (S,)
    adduct_ids = batch["adduct"][b].detach().cpu().numpy()     # (S,) per-spectrum adducts
    valid_s = np.where(batch["spec_mask"][b].detach().cpu().numpy())[0]

    # Precursor m/z is anchored at peak index 0 of each spectrum in MultiSpectrumMoleculeDataset.
    # Each spectrum keeps its own adduct (the PDF copied the primary adduct onto every spectrum,
    # so a molecule measured as [M+H]+ and [M+Na]+ produced a 22 Da disagreement).
    prec_mzs = [float(mzs_np[s, 0]) for s in valid_s]
    primary = str(batch["primary_adduct"][b])
    adducts = [INV_ADDUCT_VOCAB.get(int(adduct_ids[s]), primary) for s in valid_s]
    adducts = [primary if a == "UNKNOWN" else a for a in adducts]
    mode = str(batch["ionization_mode"][b]) if "ionization_mode" in batch else "positive"
    bpis = [float(bpi_np[s]) for s in valid_s]

    neutral_mass = subformula_filter.infer_consensus_neutral_mass(prec_mzs, adducts, bpis, [mode] * len(valid_s))
    pooled_ints = ints_np[valid_s, 1:].reshape(-1)
    return {
        "neutral_mass": neutral_mass,
        "pooled_nls": nls_np[valid_s, 1:].reshape(-1),
        "pooled_ints": pooled_ints,
        "raw_mzs": mzs_np[valid_s, 1:].reshape(-1),
        # The isotope detector's ratio thresholds are for linear intensities; the dataset stores sqrt(I).
        "raw_ints": pooled_ints ** 2,
    }


@torch.no_grad()
def generate_formula_priors_batch(
    batch: Dict[str, torch.Tensor],
    subformula_filter: FastMISTSubformulaFilter,
    fg_71_pred_batch: Optional[torch.Tensor] = None,
    ppm_window: float = W("nb4.priors.ppm_window"),
    top_k_formulas: int = W.int("nb4.priors.top_k_formulas"),
) -> List[List[str]]:
    """
    Generates `formula_priors_batch` (List of top-3 molecular formulas per molecule in batch)
    directly from a MultiSpectrumMoleculeDataset batch and optional Mass2SMILES fg_71_pred.
    """
    fg_np = fg_71_pred_batch.detach().cpu().numpy() if fg_71_pred_batch is not None else None
    formula_priors_batch: List[List[str]] = []
    for b in range(batch["mzs"].shape[0]):
        inputs = molecule_formula_inputs(batch, b, subformula_filter)
        top_formulas = subformula_filter.score_and_select_top_k_formulas(
            fg_71_pred=fg_np[b] if fg_np is not None else None,
            ppm_window=ppm_window,
            top_k=top_k_formulas,
            **inputs,
        )
        formula_priors_batch.append(top_formulas)
    return formula_priors_batch

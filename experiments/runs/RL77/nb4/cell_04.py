# [Colab Formula Filter Cell 4] Batch Integration Hook & End-to-End Verification
@torch.no_grad()
def generate_formula_priors_batch(
    batch: Dict[str, torch.Tensor],
    subformula_filter: FastMISTSubformulaFilter,
    fg_71_pred_batch: Optional[torch.Tensor] = None,
    ppm_window: float = 15.0,
    top_k_formulas: int = 3,
) -> List[List[str]]:
    """
    Generates `formula_priors_batch` (List of top-3 molecular formulas per molecule in batch)
    directly from a MultiSpectrumMoleculeDataset batch and optional Mass2SMILES fg_71_pred.
    """
    mzs_np = batch["mzs"].detach().cpu().numpy()          # (B, S, P)
    ints_np = batch["ints"].detach().cpu().numpy()        # (B, S, P)
    nls_np = batch["nls"].detach().cpu().numpy()          # (B, S, P)
    bpi_np = (10.0 ** batch["bpi"].detach().cpu().numpy()) # (B, S)
    spec_mask_np = batch["spec_mask"].detach().cpu().numpy() # (B, S)

    fg_np = (
        fg_71_pred_batch.detach().cpu().numpy()
        if fg_71_pred_batch is not None
        else None
    )

    B, S, _ = mzs_np.shape
    formula_priors_batch: List[List[str]] = []

    for b in range(B):
        valid_s = np.where(spec_mask_np[b])[0]
        # Precursor m/z is anchored at peak index 0 of each spectrum in MultiSpectrumMoleculeDataset
        prec_mzs = [float(mzs_np[b, s, 0]) for s in valid_s]
        primary_add = str(batch["primary_adduct"][b])
        adducts = [primary_add] * len(valid_s)
        bpis = [float(bpi_np[b, s]) for s in valid_s]

        neutral_mass = subformula_filter.infer_consensus_neutral_mass(
            prec_mzs, adducts, bpis
        )

        # Pool neutral losses and fragment m/z values across all valid spectra of molecule b
        pooled_nls = nls_np[b, valid_s, 1:].reshape(-1)
        pooled_ints = ints_np[b, valid_s, 1:].reshape(-1)
        pooled_mzs = mzs_np[b, valid_s, 1:].reshape(-1)

        fg_vec = fg_np[b] if fg_np is not None else None

        top_formulas = subformula_filter.score_and_select_top_k_formulas(
            neutral_mass=neutral_mass,
            pooled_nls=pooled_nls,
            pooled_ints=pooled_ints,
            raw_mzs=pooled_mzs,
            raw_ints=pooled_ints,
            fg_71_pred=fg_vec,
            ppm_window=ppm_window,
            top_k=top_k_formulas,
        )
        formula_priors_batch.append(top_formulas)

    return formula_priors_batch


if __name__ == "__main__":
    # Smoke test: Distinguish Chrysin (C15H10O4, exact mass = 254.057909 Da) from close-mass N/S isomers
    sample_catalog_formulas = [
        "C15H10O4",      # Chrysin / Daidzein (254.057909 Da) - True compound
        "C11H14N2O3S",   # Sulfonamide isomer (254.072513 Da)
        "C13H10N4O2",    # Nitrogen-rich heterocycle (254.080376 Da)
        "C14H10N2O3",    # Close-mass amide (254.069142 Da)
        "C12H11ClO4",    # Chlorinated lactone (254.034586 Da)
    ]
    adduct_shifts = {"[M+H]+": -1.007276, "[M-H]-": 1.007276}

    bank = CandidateFormulaBank(sample_catalog_formulas)
    mist_filter = FastMISTSubformulaFilter(bank, adduct_shifts, ppm_sigma=5.0)

    # Simulate a molecule_id with 2 collision-energy spectra of [M+H]+ at m/z = 255.0652 Da
    # exhibiting diagnostic neutral losses of H2O (18.0106 Da), CO (27.9949 Da), and C2H2O (42.0106 Da)
    mock_batch = {
        "mzs": torch.tensor(
            [[[255.0652, 237.0546, 227.0703, 213.0546], [255.0651, 227.0702, 183.0804, 153.0182]]],
            dtype=torch.float32,
        ),
        "ints": torch.tensor(
            [[[2.0, 0.65, 0.85, 0.50], [2.0, 0.90, 0.40, 0.95]]],
            dtype=torch.float32,
        ),
        "nls": torch.tensor(
            [[[0.0, 18.0106, 27.9949, 42.0106], [0.0, 27.9949, 71.9847, 102.0470]]],
            dtype=torch.float32,
        ),
        "bpi": torch.tensor([[4.5, 4.8]], dtype=torch.float32),
        "spec_mask": torch.tensor([[True, True]], dtype=torch.bool),
        "primary_adduct": ["[M+H]+"],
    }

    # Simulated Mass2SMILES 71-D output indicating N=0, O=4, S=0, aromatic_rings=3
    mock_fg_71 = torch.zeros((1, 71), dtype=torch.float32)
    mock_fg_71[0, 61] = math.log1p(4.0)  # Index 61 = Oxygen count (4)
    mock_fg_71[0, 69] = math.log1p(3.0)  # Index 69 = Aromatic rings (3)

    formula_priors = generate_formula_priors_batch(
        mock_batch, mist_filter, fg_71_pred_batch=mock_fg_71, ppm_window=60.0, top_k_formulas=3
    )
    print("Top-3 Predicted Formula Priors for molecule_0:", formula_priors[0])
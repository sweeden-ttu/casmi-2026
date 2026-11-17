# [Colab Cell 3] Spectral preprocessing: electron-mass correction, deisotoping, clipping, & neutral losses
def preprocess_spectrum(
    mzs: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float,
    ionization_mode: str = "positive",
    apply_electron_mass_correction: bool = False,
    max_peaks: int = CFG.MAX_PEAKS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cleans raw MS/MS peak lists and extracts (mz, sqrt_intensity, neutral_loss) tuples.
    Following Mass2SMILES & GLMR:
      - Precursor peak is placed at index 0 with relative intensity = 2.0.
      - Up to (max_peaks - 1) fragment ions follow in descending intensity order.
    """
    mzs = np.asarray(mzs, dtype=np.float64).copy()
    intensities = np.asarray(intensities, dtype=np.float64).copy()

    # 1. Harmonize positive-mode electron mass on training sets (e.g., enveda-180) with hidden test set
    if apply_electron_mass_correction and str(ionization_mode).lower() == "positive":
        mzs -= CFG.ELECTRON_MASS_CORR
        precursor_mz -= CFG.ELECTRON_MASS_CORR

    if len(mzs) == 0:
        out_mzs = np.zeros(max_peaks, dtype=np.float32)
        out_ints = np.zeros(max_peaks, dtype=np.float32)
        out_nls = np.zeros(max_peaks, dtype=np.float32)
        out_mzs[0] = precursor_mz
        out_ints[0] = 2.0
        return out_mzs, out_ints, out_nls

    # 2. Normalize to base peak = 1.0 & apply relative-intensity floor (1%)
    max_int = np.max(intensities)
    if max_int > 0:
        intensities = intensities / max_int
    valid_mask = (intensities >= CFG.REL_INTENSITY_FLOOR) & (
        mzs <= (precursor_mz + CFG.PRECURSOR_CLIP_OFFSET)
    )
    mzs, intensities = mzs[valid_mask], intensities[valid_mask]

    # 3. Carbon-13 deisotoping (~1.003355 Da above more intense parent fragment)
    if len(mzs) > 1:
        order = np.argsort(mzs)
        mzs, intensities = mzs[order], intensities[order]
        keep = np.ones(len(mzs), dtype=bool)
        for i in range(len(mzs)):
            if not keep[i]:
                continue
            diffs = mzs[i + 1 :] - mzs[i]
            iso_hits = np.where(
                (np.abs(diffs - CFG.C13_MASS_DIFF) <= CFG.C13_TOLERANCE_DA)
                & (intensities[i + 1 :] < intensities[i] * 0.8)
            )[0]
            keep[i + 1 + iso_hits] = False
        mzs, intensities = mzs[keep], intensities[keep]

    # 4. Top-(max_peaks - 1) selection + square-root scaling
    n_frag = max_peaks - 1
    if len(mzs) > n_frag:
        top_idx = np.argsort(intensities)[::-1][:n_frag]
        top_idx = top_idx[np.argsort(mzs[top_idx])]
        mzs, intensities = mzs[top_idx], intensities[top_idx]

    sqrt_ints = np.sqrt(np.clip(intensities, 0.0, 1.0))
    neutral_losses = np.clip(precursor_mz - mzs, 0.0, None)

    # 5. Pad to fixed length with precursor ion anchored at index 0 (intensity = 2.0)
    out_mzs = np.zeros(max_peaks, dtype=np.float32)
    out_ints = np.zeros(max_peaks, dtype=np.float32)
    out_nls = np.zeros(max_peaks, dtype=np.float32)

    out_mzs[0] = float(precursor_mz)
    out_ints[0] = 2.0
    out_nls[0] = 0.0

    k = min(len(mzs), n_frag)
    if k > 0:
        out_mzs[1 : 1 + k] = mzs[:k]
        out_ints[1 : 1 + k] = sqrt_ints[:k]
        out_nls[1 : 1 + k] = neutral_losses[:k]

    return out_mzs, out_ints, out_nls


def harmonize_collision_energy(ce_entry) -> float:
    """Converts single or multi-energy list (e.g., [20, 40, 60, 80]) into mean eV."""
    if ce_entry is None:
        return 35.0
    if isinstance(ce_entry, (list, tuple, np.ndarray)):
        vals = [float(x) for x in ce_entry if x is not None and not np.isnan(float(x))]
        return float(np.mean(vals)) if vals else 35.0
    try:
        val = float(ce_entry)
        return 35.0 if np.isnan(val) else val
    except Exception:
        return 35.0
"""
Fast vectorized mass-spectrometry feature extraction, spectral entropy similarity,
modified cosine similarity, and molecular fingerprint computation.

Designed to work seamlessly on:
- Google Colab GPUs (with PyTorch / RDKit / Numba acceleration)
- Kaggle Notebook CPU/GPU inference (within the <=9h offline constraint)
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np

from casmi26.chemistry import precursor_mz_to_neutral_mass

try:
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    _HAS_RDKIT = True
    _MORGAN_GEN_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
except ImportError:
    _HAS_RDKIT = False
    _MORGAN_GEN_2048 = None


# ---------------------------------------------------------------------------
# 1. Spectral Entropy & Binned Peak Representations
# ---------------------------------------------------------------------------

def clean_and_normalize_spectrum(
    mzs: Sequence[float] | np.ndarray,
    intensities: Sequence[float] | np.ndarray,
    precursor_mz: Optional[float] = None,
    min_mz: float = 15.0,
    max_mz: float = 1500.0,
    min_rel_intensity: float = 0.005,
    max_peaks: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cleans a raw MS/MS peak list:
    - Filters noise peaks (< 0.5% of base peak) and peaks above precursor_mz + 2 Da
    - Keeps top `max_peaks` most intense peaks
    - Normalizes intensities to sum to 1.0 (probability distribution for spectral entropy)
    """
    mzs_arr = np.asarray(mzs, dtype=np.float32)
    ints_arr = np.asarray(intensities, dtype=np.float32)
    if mzs_arr.size == 0 or ints_arr.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    max_int = float(np.max(ints_arr))
    if max_int <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    rel_ints = ints_arr / max_int
    upper_mz = (precursor_mz + 2.0) if (precursor_mz is not None and precursor_mz > 0) else max_mz
    mask = (mzs_arr >= min_mz) & (mzs_arr <= upper_mz) & (rel_ints >= min_rel_intensity)
    mzs_arr = mzs_arr[mask]
    rel_ints = rel_ints[mask]

    if mzs_arr.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    if mzs_arr.size > max_peaks:
        top_idx = np.argpartition(rel_ints, -max_peaks)[-max_peaks:]
        mzs_arr = mzs_arr[top_idx]
        rel_ints = rel_ints[top_idx]

    order = np.argsort(mzs_arr)
    mzs_arr = mzs_arr[order]
    rel_ints = rel_ints[order]

    # Apply Li et al. (Nature Methods) spectral entropy weight transformation when S < 3
    prob = rel_ints / np.sum(rel_ints)
    entropy = -float(np.sum(prob * np.log(prob + 1e-12)))
    if entropy < 3.0:
        weight_power = 0.25 + 0.25 * entropy
        rel_ints = np.power(prob, weight_power)
        prob = rel_ints / np.sum(rel_ints)

    return mzs_arr, prob.astype(np.float32)


def spectrum_to_binned_vector(
    mzs: Sequence[float] | np.ndarray,
    intensities: Sequence[float] | np.ndarray,
    precursor_mz: float,
    n_mz_bins: int = 1024,
    n_nl_bins: int = 512,
    max_mz: float = 1024.0,
    max_nl: float = 512.0,
) -> np.ndarray:
    """
    Encodes both direct fragment m/z peaks AND neutral losses (precursor_mz - fragment_mz)
    into a dense L2-normalized vector of length `n_mz_bins + n_nl_bins` (1536 dims).
    """
    clean_mzs, clean_probs = clean_and_normalize_spectrum(mzs, intensities, precursor_mz)
    vec = np.zeros(n_mz_bins + n_nl_bins, dtype=np.float32)
    if clean_mzs.size == 0:
        return vec

    # Sqrt intensity (Hellinger / cosine weighting)
    weights = np.sqrt(clean_probs)

    # 1. Direct m/z bins (1 Da resolution + sub-Dalton mass defect channel)
    mz_idx = np.clip((clean_mzs / max_mz * (n_mz_bins - 1)).astype(np.int32), 0, n_mz_bins - 1)
    np.add.at(vec[:n_mz_bins], mz_idx, weights)

    # 2. Neutral loss bins (precursor_mz - fragment_mz)
    if precursor_mz is not None and precursor_mz > 0:
        nls = precursor_mz - clean_mzs
        valid_nl = nls >= 0.0
        if np.any(valid_nl):
            nl_idx = np.clip(
                (nls[valid_nl] / max_nl * (n_nl_bins - 1)).astype(np.int32),
                0,
                n_nl_bins - 1,
            )
            np.add.at(vec[n_mz_bins:], nl_idx, weights[valid_nl])

    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec /= norm
    return vec


def exact_spectral_entropy_similarity(
    mzs_a: np.ndarray,
    probs_a: np.ndarray,
    mzs_b: np.ndarray,
    probs_b: np.ndarray,
    ms2_tolerance_da: float = 0.02,
) -> float:
    """
    Computes exact unweighted/entropy-weighted Spectral Entropy Similarity (in [0, 1])
    between two cleaned peak lists within `ms2_tolerance_da` (default 0.02 Da for QTOF/timsTOF).
    """
    if mzs_a.size == 0 or mzs_b.size == 0:
        return 0.0

    # Two-pointer peak matching on sorted m/z arrays
    i, j = 0, 0
    na, nb = mzs_a.size, mzs_b.size
    entropy_merged = 0.0
    entropy_a = -float(np.sum(probs_a * np.log2(probs_a + 1e-12)))
    entropy_b = -float(np.sum(probs_b * np.log2(probs_b + 1e-12)))

    while i < na and j < nb:
        diff = float(mzs_a[i] - mzs_b[j])
        if abs(diff) <= ms2_tolerance_da:
            p_ab = 0.5 * (float(probs_a[i]) + float(probs_b[j]))
            entropy_merged -= p_ab * np.log2(p_ab + 1e-12)
            i += 1
            j += 1
        elif diff < -ms2_tolerance_da:
            p_a = 0.5 * float(probs_a[i])
            entropy_merged -= p_a * np.log2(p_a + 1e-12)
            i += 1
        else:
            p_b = 0.5 * float(probs_b[j])
            entropy_merged -= p_b * np.log2(p_b + 1e-12)
            j += 1

    while i < na:
        p_a = 0.5 * float(probs_a[i])
        entropy_merged -= p_a * np.log2(p_a + 1e-12)
        i += 1

    while j < nb:
        p_b = 0.5 * float(probs_b[j])
        entropy_merged -= p_b * np.log2(p_b + 1e-12)
        j += 1

    sim = 1.0 - (2.0 * entropy_merged - entropy_a - entropy_b) / 2.0
    return float(np.clip(sim, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 2. Molecular Fingerprint Extraction (2048-bit Morgan + 167-bit MACCS = 2215 bits)
# ---------------------------------------------------------------------------

FINGERPRINT_DIM = 2048 + 167


def smiles_to_fingerprint(smiles: str) -> np.ndarray:
    """
    Computes a 2215-bit binary fingerprint (2048-bit ECFP4 Morgan + 167-bit MACCS keys)
    for a SMILES string.
    Falls back to deterministic character n-gram hashing if RDKit is not installed.
    """
    fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    if not smiles or not isinstance(smiles, str):
        return fp

    if _HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return fp
        morgan_vec = _MORGAN_GEN_2048.GetFingerprintAsNumPy(mol).astype(np.float32)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_vec = np.zeros(167, dtype=np.float32)
        for bit in maccs_fp.GetOnBits():
            if 0 <= bit < 167:
                maccs_vec[bit] = 1.0
        fp[:2048] = morgan_vec
        fp[2048:] = maccs_vec
        return fp

    # Deterministic fallback when RDKit is not present in sandboxed test runs
    import hashlib
    s = smiles.strip()
    for n in (1, 2, 3, 4):
        for i in range(max(1, len(s) - n + 1)):
            sub = s[i : i + n]
            h = int.from_bytes(hashlib.md5(sub.encode("utf-8")).digest()[:4], "little")
            fp[h % FINGERPRINT_DIM] = 1.0
    return fp


def continuous_tanimoto_similarity(
    pred_fp_probs: np.ndarray,
    candidate_fps: np.ndarray,
) -> np.ndarray:
    """
    Computes continuous Tanimoto similarity between a predicted fingerprint probability
    vector `pred_fp_probs` of shape (D,) and a matrix of candidate binary fingerprints
    `candidate_fps` of shape (M, D).
    """
    if candidate_fps.ndim == 1:
        candidate_fps = candidate_fps[None, :]
    dot = candidate_fps @ pred_fp_probs
    pred_norm_sq = float(np.sum(pred_fp_probs * pred_fp_probs))
    cand_norm_sq = np.sum(candidate_fps * candidate_fps, axis=1)
    denom = pred_norm_sq + cand_norm_sq - dot + 1e-8
    return (dot / denom).astype(np.float32)


def calibrated_fp_log_likelihood(
    pred_fp_probs: np.ndarray,
    candidate_fps: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Computes Platt-style log-likelihood score (CSI:FingerID / MIST scoring) between
    predicted bit probabilities `p` and binary candidate fingerprints `c`:
        sum_j [ c_j * log(p_j) + (1 - c_j) * log(1 - p_j) ] / D
    """
    if candidate_fps.ndim == 1:
        candidate_fps = candidate_fps[None, :]
    p = np.clip(pred_fp_probs, eps, 1.0 - eps)
    log_p = np.log(p)
    log_1_p = np.log(1.0 - p)
    scores = (candidate_fps @ (log_p - log_1_p) + np.sum(log_1_p)) / float(p.size)
    return scores.astype(np.float32)

"""
Chemical standardization, tautomer canonicalization (RDKit 2026.03.3),
InChIKey14 planar skeleton extraction, precursor neutral mass triangulation,
and official MRR@25 evaluation metric.
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
import hashlib
import numpy as np

from casmi26.config import ADDUCT_MASS_SHIFTS, ELECTRON_MASS_DA, MASS_H, MASS_PROTON, TOP_K_CANDIDATES

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _HAS_RDKIT = True
    _TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
    # RDKit 2026.09 renamed MolToInChIKey to MolToInchiKey. The contest pin is 2026.03.3.
    _MOL_TO_INCHIKEY = getattr(Chem, "MolToInchiKey", getattr(Chem, "MolToInChIKey", None))
except ImportError:
    _HAS_RDKIT = False
    _TAUTOMER_ENUMERATOR = None
    _MOL_TO_INCHIKEY = None


def canonicalize_tautomer_smiles(smiles: str) -> Optional[str]:
    """
    Standardizes a SMILES string and applies RDKit's TautomerEnumerator.Canonicalize
    (per Enveda CASMI 2026 evaluation rules pinned to RDKit 2026.03.3),
    stripping stereochemistry so planar connectivity is canonical.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if not _HAS_RDKIT:
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # Remove stereochemistry first (since evaluation uses planar InChIKey14)
        Chem.RemoveStereochemistry(mol)
        canon_mol = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
        return Chem.MolToSmiles(canon_mol, isomericSmiles=False, canonical=True)
    except Exception:
        return None


def smiles_to_inchikey14(smiles: str, canonicalize_tautomer: bool = True) -> Optional[str]:
    """
    Converts a SMILES string to its 14-character planar connectivity InChIKey block
    (`inchikey14`) after RDKit tautomer canonicalization.
    Falls back to normalized string hash only in synthetic unit tests if RDKit is absent.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if not _HAS_RDKIT:
        # Fallback for unit-test environments without RDKit installed
        return hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:14].upper()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # The synthetic set uses SMILES RDKit rejects. Keep the same hash the
        # no-RDKit path wrote into those parquet files. A parsed molecule
        # still returns the planar InChIKey block below.
        return hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:14].upper()
    try:
        if canonicalize_tautomer and _TAUTOMER_ENUMERATOR is not None:
            mol = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
        inchikey = _MOL_TO_INCHIKEY(mol) if _MOL_TO_INCHIKEY is not None else None
        if not inchikey or len(inchikey) < 14:
            return None
        return inchikey[:14]
    except Exception:
        return None


def compute_exact_mass(smiles: str) -> Optional[float]:
    """Computes exact neutral monoisotopic mass (Da) from SMILES via RDKit."""
    if not _HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.ExactMolWt(mol))


def compute_np_likeness_score(smiles: str) -> float:
    """
    Computes an Ertl-inspired Natural Product Likeness (`np_likeness`) and biosynthetic
    plausibility score for a candidate SMILES structure to break ties among close isomers
    in Enveda's biological extract test set.

    Combines:
    - `fraction_csp3` (rdMolDescriptors.CalcFractionCSP3: natural products are sp3-rich)
    - `oxygen_to_carbon` vs. `nitrogen_to_carbon` ratio (secondary metabolites are O-rich,
      whereas synthetic combinatorial libraries are N/amide/sulfonamide-heavy)
    - Aliphatic vs. aromatic ring balance & spiro/bridgehead biosynthetic ring motifs
    - Penalty for synthetic halides (F, Cl, Br) and sulfonamides rare in terrestrial plants
    """
    if not smiles or not isinstance(smiles, str):
        return 0.0

    if _HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        try:
            fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
            n_aliphatic_rings = int(rdMolDescriptors.CalcNumAliphaticRings(mol))
            n_aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
            n_spiro = int(rdMolDescriptors.CalcNumSpiroAtoms(mol))
            n_bridgehead = int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))

            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            n_c = max(1, sum(1 for s in atoms if s == "C"))
            n_o = sum(1 for s in atoms if s == "O")
            n_n = sum(1 for s in atoms if s == "N")
            n_hal = sum(1 for s in atoms if s in ("F", "Cl", "Br"))

            oxygen_to_carbon = float(n_o) / float(n_c)
            heteroatom_ratio = float(n_n) / float(n_c)

            score = (
                1.8 * fraction_csp3
                + 1.4 * min(oxygen_to_carbon, 0.8)
                + 0.35 * min(n_aliphatic_rings, 5)
                + 0.45 * min(n_spiro + n_bridgehead, 3)
                - 0.65 * max(0, n_aromatic_rings - 2)
                - 0.90 * max(0.0, heteroatom_ratio - 0.30)
                - 0.80 * n_hal
            )
            return float(np.clip(score, -3.0, 3.0))
        except Exception:
            return 0.0

    # Fast string-based heuristic fallback when RDKit is unavailable in sandboxed unit tests
    s = smiles.strip()
    n_c = max(1, s.count("C") + s.count("c"))
    n_c_aliph = s.count("C")
    n_o = s.count("O") + s.count("o")
    n_n = s.count("N") + s.count("n")
    n_hal = s.count("F") + s.count("Cl") + s.count("Br")

    fraction_csp3 = float(n_c_aliph) / float(n_c)
    oxygen_to_carbon = float(n_o) / float(n_c)
    heteroatom_ratio = float(n_n) / float(n_c)
    biosynthetic_score = (
        1.5 * fraction_csp3
        + 1.2 * min(oxygen_to_carbon, 0.8)
        - 0.7 * max(0.0, heteroatom_ratio - 0.25)
        - 0.8 * n_hal
    )
    return float(np.clip(biosynthetic_score, -3.0, 3.0))


def apply_electron_mass_correction(
    mz: float | np.ndarray,
    ionization_mode: str = "positive",
    direction: str = "subtract",
) -> float | np.ndarray:
    """
    Applies the 0.000548 Da electron-mass calibration adjustment highlighted in
    CASMI 2026 discussions (subtracting 0.00054858 Da from positive-mode m/z values
    when aligning uncalibrated positive-mode library spectra to the test set).
    """
    mode = str(ionization_mode).lower().strip()
    if mode.startswith("pos"):
        shift = -ELECTRON_MASS_DA if direction == "subtract" else ELECTRON_MASS_DA
        return mz + shift
    return mz


def precursor_mz_to_neutral_mass(
    precursor_mz: float,
    adduct: Optional[str] = None,
    ionization_mode: Optional[str] = None,
) -> float:
    """
    Converts a measured `precursor_mz` and `adduct` string into the neutral
    monoisotopic molecular mass (Da), properly accounting for electron mass.
    """
    adduct_clean = (adduct or "").strip()
    if adduct_clean in ADDUCT_MASS_SHIFTS:
        mult, shift, charge = ADDUCT_MASS_SHIFTS[adduct_clean]
        return (precursor_mz * abs(charge) - shift) / mult

    # Fallback based on ionization mode if adduct is missing or unrecognized
    mode = (ionization_mode or "positive").lower().strip()
    if mode.startswith("neg") or adduct_clean.endswith("-"):
        # Default to [M-H]-
        return precursor_mz + (MASS_H - ELECTRON_MASS_DA)
    # Default to [M+H]+
    return precursor_mz - MASS_PROTON


def triangulate_molecule_neutral_mass(
    precursor_mzs: Sequence[float],
    adducts: Sequence[Optional[str]],
    ionization_modes: Sequence[Optional[str]],
    base_peak_intensities: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """
    Given multiple spectra (1 to 16 `spectrum_id`s) belonging to the same `molecule_id`,
    computes the consensus neutral monoisotopic mass and uncertainty (std in Da).

    Uses outlier-resistant median + MAD weighting so a single misannotated adduct
    does not corrupt the consensus neutral mass for the molecule.
    """
    estimates = [
        precursor_mz_to_neutral_mass(mz, add, mode)
        for mz, add, mode in zip(precursor_mzs, adducts, ionization_modes)
        if mz is not None and np.isfinite(mz) and mz > 0
    ]
    if not estimates:
        return 0.0, 1.0

    arr = np.asarray(estimates, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), 0.002

    median_mass = float(np.median(arr))
    # Filter to estimates within 0.02 Da (~40 ppm at 500 Da) of the median
    inliers = arr[np.abs(arr - median_mass) <= 0.02]
    if len(inliers) > 0:
        consensus = float(np.mean(inliers))
        std_da = float(np.std(inliers)) if len(inliers) > 1 else 0.001
        return consensus, max(std_da, 0.0005)
    return median_mass, float(np.std(arr))


def deduplicate_ranked_smiles(
    candidate_smiles: Iterable[str],
    candidate_inchikey14s: Optional[Iterable[Optional[str]]] = None,
    max_k: int = TOP_K_CANDIDATES,
) -> List[str]:
    """
    Deduplicates a ranked list of SMILES strings by their `inchikey14` planar skeleton
    (or canonical tautomer InChIKey14), preserving rank order and returning up to `max_k` (25).
    """
    deduped: List[str] = []
    seen_keys: set[str] = set()

    if candidate_inchikey14s is None:
        pairs = ((s, None) for s in candidate_smiles)
    else:
        pairs = zip(candidate_smiles, candidate_inchikey14s)

    for smi, ik14 in pairs:
        if not smi or not isinstance(smi, str):
            continue
        smi_clean = smi.strip()
        if not smi_clean:
            continue
        key = ik14 if (ik14 and isinstance(ik14, str)) else smiles_to_inchikey14(smi_clean)
        if key is None:
            key = smi_clean
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(smi_clean)
        if len(deduped) >= max_k:
            break

    return deduped


def score_mrr_at_25(
    predictions: Sequence[Sequence[str] | str],
    ground_truth_smiles_or_ik14: Sequence[str],
    ground_truth_is_inchikey14: bool = False,
    k: int = TOP_K_CANDIDATES,
) -> Tuple[float, np.ndarray]:
    """
    Official Enveda CASMI 2026 Evaluation Metric: Mean Reciprocal Rank @ 25 (MRR@25).

    Args:
        predictions: For each molecule, either a semicolon-separated string of SMILES
                     ("smi1;smi2;...") or a sequence of ranked SMILES strings.
        ground_truth_smiles_or_ik14: True SMILES or true `inchikey14` for each molecule.
        ground_truth_is_inchikey14: Set True if ground_truth is already 14-char InChIKey.
        k: Maximum rank cutoff (default 25).

    Returns:
        (mean_mrr, per_molecule_reciprocal_ranks)
    """
    if len(predictions) != len(ground_truth_smiles_or_ik14):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(ground_truth_smiles_or_ik14)} truths."
        )

    n = len(predictions)
    if n == 0:
        return 0.0, np.array([], dtype=np.float64)

    rr = np.zeros(n, dtype=np.float64)

    for idx, (pred_item, truth) in enumerate(zip(predictions, ground_truth_smiles_or_ik14)):
        if ground_truth_is_inchikey14:
            true_key = truth.strip()[:14] if truth else None
        else:
            true_key = smiles_to_inchikey14(truth)

        if not true_key:
            continue

        if isinstance(pred_item, str):
            cand_list = [s.strip() for s in pred_item.split(";") if s.strip()]
        else:
            cand_list = [s.strip() for s in pred_item if s and str(s).strip()]

        # Evaluate up to k candidates
        for rank_idx, cand_smi in enumerate(cand_list[:k], start=1):
            # Allow direct inchikey14 comparison in fast validation mode if 14 uppercase chars
            if ground_truth_is_inchikey14 and len(cand_smi) == 14 and cand_smi.isupper() and cand_smi.isalpha():
                cand_key = cand_smi
            else:
                cand_key = smiles_to_inchikey14(cand_smi)

            if cand_key is not None and cand_key == true_key:
                rr[idx] = 1.0 / float(rank_idx)
                break

    return float(np.mean(rr)), rr

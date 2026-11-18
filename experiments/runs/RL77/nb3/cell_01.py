# [Colab Ensemble Cell 1] 2000-Bin Spectral Vectorizer, ECFP4 (2048-bit), & 71-D Functional Group Extractor
import inspect
from functools import lru_cache
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Fragments, rdMolDescriptors, rdFingerprintGenerator

# Deterministically collect the first 60 RDKit functional group fragment functions (fr_*, sorted by name)
RDKIT_FG_FUNCTIONS = sorted(
    [
        (name, func)
        for name, func in inspect.getmembers(Fragments, inspect.isfunction)
        if name.startswith("fr_")
    ],
    key=lambda x: x[0],
)[:60]

HETEROATOM_SYMBOLS = ["N", "O", "S", "P", "F", "Cl", "Br", "I"]

# Approximate oxygen/nitrogen-rich pyranose/furanose glycoside SMARTS for fast sugar moiety counting
SUGAR_MOIETY_SMARTS = Chem.MolFromSmarts("[OX2r5,OX2r6]1[CX4r5,CX4r6]([OX2H1,OX2])[CX4r5,CX4r6]([OX2H1,OX2])[CX4r5,CX4r6]([OX2H1,OX2])[CX4r5,CX4r6]1")


def spectrum_to_2000_bins(
    mzs: np.ndarray,
    intensities: np.ndarray,
    min_mz: float = 50.0,
    max_mz: float = 1250.0,
    num_bins: int = 2000,
) -> np.ndarray:
    """
    Bins a cleaned MS/MS peak list into a 2000-dimensional normalized intensity vector
    (Carrera et al., BIOSTEC 2026 Section 3.3).
    """
    binned = np.zeros(num_bins, dtype=np.float32)
    if len(mzs) == 0:
        return binned

    mzs = np.asarray(mzs, dtype=np.float32)
    intensities = np.asarray(intensities, dtype=np.float32)
    max_i = np.max(intensities)
    if max_i > 0:
        intensities = intensities / max_i

    valid = (mzs >= min_mz) & (mzs < max_mz) & (intensities >= 0.005)
    mzs, intensities = mzs[valid], np.sqrt(intensities[valid])

    bin_indices = ((mzs - min_mz) / (max_mz - min_mz) * num_bins).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Max-pool intensities within each mass bin
    np.maximum.at(binned, bin_indices, intensities)
    norm = np.linalg.norm(binned)
    if norm > 0:
        binned = binned / norm
    return binned


def aggregate_molecule_2000_bins(
    spectra_records: List[ dict ],
    num_bins: int = 2000,
) -> np.ndarray:
    """
    Pools multiple collision-energy spectra for a single molecule_id into a composite
    2000-bin vector weighted by log10(base_peak_intensity).
    """
    if not spectra_records:
        return np.zeros(num_bins, dtype=np.float32)

    composite = np.zeros(num_bins, dtype=np.float32)
    total_weight = 0.0
    for row in spectra_records:
        vec = spectrum_to_2000_bins(
            row["ms2_mzs"], row["ms2_normalized_intensities"], num_bins=num_bins
        )
        bpi = row.get("base_peak_intensity", 1e4)
        bpi = float(bpi) if bpi is not None and np.isfinite(float(bpi)) else 1e4  # 46% of train values are null
        bpi = max(bpi, 10.0)
        w = float(np.log10(bpi))
        composite += w * vec
        total_weight += w

    if total_weight > 0:
        composite /= total_weight
    norm = np.linalg.norm(composite)
    return composite / norm if norm > 0 else composite


@lru_cache(maxsize=8)
def _morgan_generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def smiles_to_ecfp4_and_mass2smiles_71d(
    smiles: str,
    n_bits: int = 2048,
    radius: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extracts:
      1. 2048-bit Morgan (ECFP4, radius=2) binary fingerprint (Carrera et al.).
      2. 71-dimensional scaled structural vector (Elser et al., Mass2SMILES):
         - Indices 0..59:  60 RDKit functional group counts
         - Indices 60..67: 8 heteroatom counts (N, O, S, P, F, Cl, Br, I)
         - Indices 68..70: 3 sugar/ring counts (sugar rings, aromatic rings, aliphatic rings)
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None, None

    # 1. 2048-bit ECFP4 fingerprint
    # (AllChem.GetMorganFingerprintAsBitVect is deprecated in current RDKit; the generator gives the same bits.)
    ecfp4_arr = _morgan_generator(radius, n_bits).GetFingerprintAsNumPy(mol).astype(np.float32)

    # 2. Mass2SMILES 71-d functional group + heteroatom + sugar descriptor vector
    fg_71 = np.zeros(71, dtype=np.float32)
    for idx, (_, fg_fn) in enumerate(RDKIT_FG_FUNCTIONS):
        try:
            fg_71[idx] = float(fg_fn(mol))
        except Exception:
            fg_71[idx] = 0.0

    # Heteroatom counts (indices 60..67)
    atom_counts = {sym: 0 for sym in HETEROATOM_SYMBOLS}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in atom_counts:
            atom_counts[sym] += 1
    for j, sym in enumerate(HETEROATOM_SYMBOLS):
        fg_71[60 + j] = float(atom_counts[sym])

    # Sugar moiety count & ring counts (indices 68..70)
    sugar_hits = len(mol.GetSubstructMatches(SUGAR_MOIETY_SMARTS)) if SUGAR_MOIETY_SMARTS else 0
    fg_71[68] = float(sugar_hits)
    fg_71[69] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    fg_71[70] = float(rdMolDescriptors.CalcNumAliphaticRings(mol))

    # Log1p scaling stabilizes regression across high-count functional groups & heteroatoms
    fg_71_scaled = np.log1p(np.clip(fg_71, 0.0, None))
    return ecfp4_arr, fg_71_scaled
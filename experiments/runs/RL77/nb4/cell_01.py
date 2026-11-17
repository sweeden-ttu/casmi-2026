# [Colab Formula Filter Cell 1] Exact IUPAC masses, Hill-notation stoichiometry parser, and diagnostic MS/MS neutral losses
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch

# Ordered 10-element stoichiometry vector: [C, H, N, O, P, S, F, Cl, Br, I]
ELEMENTS = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"]
ELEM_INDEX = {el: i for i, el in enumerate(ELEMENTS)}

# Exact monoisotopic atomic masses (IUPAC / NIST)
EXACT_ATOMIC_MASSES = np.array(
    [
        12.0000000,   # C
        1.007825032,  # H
        14.003074004, # N
        15.994914620, # O
        30.973761998, # P
        31.972071174, # S
        18.998403163, # F
        34.968852682, # Cl
        78.918337600, # Br
        126.90447190, # I
    ],
    dtype=np.float64,
)

# Standard valences for Ring & Double Bond Equivalent (RDBE) calculation:
# RDBE = C - 0.5*(H + F + Cl + Br + I) + 0.5*(N + P) + 1
RDBE_COEFFS = np.array(
    [1.0, -0.5, 0.5, 0.0, 0.5, 0.0, -0.5, -0.5, -0.5, -0.5], dtype=np.float64
)

FORMULA_REGEX = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula_to_vector(formula: str) -> Optional[np.ndarray]:
    """
    Converts a Hill-notation molecular formula string (e.g., 'C15H10O4') into a
    10-element integer stoichiometry vector [C, H, N, O, P, S, F, Cl, Br, I].
    Returns None if the formula contains unsupported metallic/organometallic elements.
    """
    vec = np.zeros(len(ELEMENTS), dtype=np.int16)
    pos = 0
    for match in FORMULA_REGEX.finditer(formula):
        if match.start() != pos:
            return None
        pos = match.end()
        elem, count_str = match.groups()
        if elem not in ELEM_INDEX:
            return None
        count = int(count_str) if count_str else 1
        vec[ELEM_INDEX[elem]] += count
    if pos != len(formula) or vec[0] <= 0:
        return None
    return vec


def vector_to_hill_formula(vec: np.ndarray) -> str:
    """Formats a 10-element stoichiometry vector into standard Hill notation (C, H, then alphabetical)."""
    parts = []
    # Carbon first, Hydrogen second
    for idx in (0, 1):
        c = int(vec[idx])
        if c > 0:
            parts.append(f"{ELEMENTS[idx]}{c if c > 1 else ''}")
    # Remaining elements in alphabetical order: Br, Cl, F, I, N, O, P, S
    alpha_order = sorted(range(2, len(ELEMENTS)), key=lambda i: ELEMENTS[i])
    for idx in alpha_order:
        c = int(vec[idx])
        if c > 0:
            parts.append(f"{ELEMENTS[idx]}{c if c > 1 else ''}")
    return "".join(parts)


# 36 high-frequency diagnostic neutral losses in timsTOF LC-MS/MS spectra:
# (Name, Stoichiometry String) -> converted to exact mass & required atom counts
DIAGNOSTIC_NEUTRAL_LOSS_SPECS = [
    ("H2", "H2"),
    ("CH4", "CH4"),
    ("NH3", "H3N"),
    ("H2O", "H2O"),
    ("HF", "FH"),
    ("HCN", "CHN"),
    ("CO", "CO"),
    ("C2H4", "C2H4"),
    ("CH3N", "CH3N"),
    ("H2CO", "CH2O"),
    ("CH3OH", "CH4O"),
    ("H2S", "H2S"),
    ("HCl", "ClH"),
    ("C2H2O", "C2H2O"),
    ("C3H6", "C3H6"),
    ("CHNO", "CHNO"),
    ("CO2", "CO2"),
    ("CH3NO", "CH3NO"),
    ("C2H5N", "C2H5N"),
    ("CH2O2", "CH2O2"),
    ("C2H6O", "C2H6O"),
    ("NO2", "NO2"),
    ("CH4S", "CH4S"),
    ("C3H4O", "C3H4O"),
    ("C2H4O2", "C2H4O2"),
    ("CH4N2O", "CH4N2O"),
    ("SO2", "O2S"),
    ("C5H8", "C5H8"),          # Isoprene unit (terpenoids)
    ("HBr", "BrH"),
    ("HPO3", "HO3P"),          # Phosphate ester loss
    ("SO3", "O3S"),            # Sulfate ester loss
    ("C3H6O3", "C3H6O3"),
    ("H3PO4", "H3O4P"),        # Phosphoric acid loss
    ("HI", "HI"),
    ("C5H8O4", "C5H8O4"),      # Pentose / Arabinose / Xylose glycoside loss
    ("C6H10O4", "C6H10O4"),    # Deoxyhexose / Rhamnose glycoside loss
    ("C6H10O5", "C6H10O5"),    # Hexose / Glucose / Galactose glycoside loss
    ("C6H8O6", "C6H8O6"),      # Glucuronic acid loss
]


def _compile_diagnostic_loss_tables() -> Tuple[np.ndarray, np.ndarray]:
    vecs = []
    for _, form_str in DIAGNOSTIC_NEUTRAL_LOSS_SPECS:
        # Temporarily parse without requiring Carbon > 0
        v = np.zeros(len(ELEMENTS), dtype=np.int16)
        for m in FORMULA_REGEX.finditer(form_str):
            el, cnt = m.groups()
            v[ELEM_INDEX[el]] += int(cnt) if cnt else 1
        vecs.append(v)
    nl_atom_matrix = np.stack(vecs, axis=0)  # (N_nl, 10)
    nl_exact_masses = nl_atom_matrix.astype(np.float64) @ EXACT_ATOMIC_MASSES
    order = np.argsort(nl_exact_masses)
    return nl_exact_masses[order], nl_atom_matrix[order]


NL_EXACT_MASSES, NL_ATOM_MATRIX = _compile_diagnostic_loss_tables()
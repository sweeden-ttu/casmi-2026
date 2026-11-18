"""Shared pieces for the foundation-model experiments (EN1, EN4, RR1, IV1, DN6).

The experiments are judged on RL77's real benchmark, not on `data/synthetic`:
  * validation = notebook 1's 998 held-out timsTOF molecules (`RL77/build/outputs/nb1/split/val_inchikey14.txt`);
  * candidates = every structure of the notebook-2 store (727,923 structures) inside the molecule's mass window, with
    RL77's adduct table and window rule (15 ppm, at least 0.015 Da);
  * answer key = InChIKey14 of the tautomer-canonical true structure (RL77 `ground_truth_ik14`).

This module is imported both locally (export, scoring) and on the Colab VM (it only needs numpy/pandas/pyarrow).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

FOUNDATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = FOUNDATION_DIR.parents[1]
INPUTS_DIR = FOUNDATION_DIR / "inputs"
RL77 = REPO_ROOT / "experiments" / "runs" / "RL77"
SEED = 3407

PPM_TOL = 15.0          # RL77 W("nb2.retrieval.ppm_tol")
MIN_TOL_DA = 0.015      # RL77 W("nb2.retrieval.min_tolerance_da")
POSITIVE_MONOMER = ("[M+H]+", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M-H2O+H]+", "[M]+")  # what MIST/DiffMS/DualLGD accept


def rl77_adduct_specs(cell: Path = RL77 / "nb1" / "cell_02.py") -> Dict[str, Tuple[int, int, float]]:
    """RL77's ADDUCT_SPECS (n, z, delta), evaluated from notebook 1's cell so the two never drift apart."""
    tree = ast.parse(cell.read_text(encoding="utf-8"))
    wanted = {"PROTON", "ELECTRON", "ADDUCT_SPECS"}
    body = [n for n in tree.body
            if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets))
            or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in wanted)]
    namespace: Dict = {"Dict": Dict, "Tuple": Tuple}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(cell), "exec"), namespace)
    return namespace["ADDUCT_SPECS"]


ADDUCT_SPECS = rl77_adduct_specs() if (RL77 / "nb1" / "cell_02.py").exists() else {}


def neutral_mass(precursor_mz: float, adduct: str, ionization_mode: str = "positive") -> float:
    """RL77 `neutral_mass_from_precursor`: M = (|z| * m/z - delta) / n, unknown adducts fall back to [M+H]+ / [M-H]-."""
    spec = ADDUCT_SPECS.get(str(adduct))
    if spec is None:
        spec = ADDUCT_SPECS["[M-H]-"] if str(ionization_mode).lower().startswith("neg") else ADDUCT_SPECS["[M+H]+"]
    n, z, delta = spec
    return (abs(z) * float(precursor_mz) - delta) / n


def primary_index(adducts: Iterable[str]) -> int:
    """RL77 `_primary_record_index`: first spectrum with a singly charged monomer adduct."""
    for i, adduct in enumerate(adducts):
        spec = ADDUCT_SPECS.get(str(adduct))
        if spec is not None and spec[0] == 1 and abs(spec[1]) == 1:
            return i
    return 0


def window_bounds(masses_sorted: np.ndarray, mass: float) -> Tuple[int, int]:
    tol = max(mass * PPM_TOL * 1e-6, MIN_TOL_DA)
    return int(np.searchsorted(masses_sorted, mass - tol, side="left")), int(np.searchsorted(masses_sorted, mass + tol, side="right"))


def mrr25_by_molecule(scores: pd.DataFrame, truth: Mapping[str, str], score_col: str = "score") -> pd.Series:
    """Reciprocal rank (top 25, first match) of the truth InChIKey14 per molecule.

    `scores` has one row per (molecule_id, inchikey14) candidate. Missing scores rank last (stable on the candidate
    order), so a stream that cannot score a molecule gives it the window order, as the blend would without it.
    """
    out = {}
    for mol_id, grp in scores.groupby("molecule_id", sort=False):
        true_ik = truth.get(mol_id)
        if true_ik is None:
            continue
        order = grp[score_col].fillna(-np.inf).to_numpy()
        ranked = grp["inchikey14"].to_numpy()[np.argsort(-order, kind="stable")]
        seen, rank, rr = set(), 0, 0.0
        for ik in ranked:
            if ik in seen:
                continue
            seen.add(ik)
            rank += 1
            if ik == true_ik:
                rr = 1.0 / rank
                break
            if rank >= 25:
                break
        out[mol_id] = rr
    return pd.Series(out, dtype=float)


def summarize(rr: pd.Series, novel_mask: Optional[pd.Series] = None) -> Dict[str, float]:
    """In-library MRR@25 plus the test-like view: molecules whose truth only the training library contains score 0,
    as they would on the leaderboard where test structures need not be in train.parquet."""
    res = {"mrr25": float(rr.mean()) if len(rr) else 0.0, "n": int(len(rr))}
    if novel_mask is not None:
        mask = novel_mask.reindex(rr.index).fillna(False).astype(bool)
        res["mrr25_test_like"] = float(rr.where(mask, 0.0).mean()) if len(rr) else 0.0
        res["truth_in_coconut"] = float(mask.mean()) if len(mask) else 0.0
    return res


def write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")


def load_inputs(inputs_dir: Path = INPUTS_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(spectra for val+test, candidates, molecules) as written by export_inputs.py."""
    spectra = pd.concat([pd.read_parquet(inputs_dir / f"{s}_spectra.parquet") for s in ("val", "test")], ignore_index=True)
    return spectra, pd.read_parquet(inputs_dir / "candidates.parquet"), pd.read_parquet(inputs_dir / "molecules.parquet")


def stream_metrics(scores: pd.DataFrame, molecules: pd.DataFrame, score_col: str = "score") -> Dict[str, Dict[str, float]]:
    """Per-stream validation metrics on the val molecules (in-library and test-like)."""
    val = molecules[molecules["split"] == "val"].set_index("molecule_id")
    rr = mrr25_by_molecule(scores[scores["molecule_id"].isin(val.index)], val["truth_ik14"].to_dict(), score_col)
    return {"all": summarize(rr, val["truth_in_coconut"]),
            "scored_only": summarize(rr[rr.index.isin(scores.loc[scores[score_col].notna(), "molecule_id"])],
                                     val["truth_in_coconut"])}


def list_column(values: List[np.ndarray]) -> List[List[float]]:
    return [np.asarray(v, dtype=np.float32).tolist() for v in values]

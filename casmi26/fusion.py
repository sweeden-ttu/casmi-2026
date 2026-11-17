"""P5 candidate merge.

Database, language-model, and diffusion candidates are stacked, scored with
the active Evidently blend, and deduplicated on (molecule_id, inchikey14).
IV4 passes DualLGD rows through this same function after the retrieval head
has filled the blend columns. Rows that only have a generator score stay
ordered by that score until the retrieval features exist.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd

SOURCE_FRAME = Tuple[str, pd.DataFrame]


def linear_blend(frame: pd.DataFrame, weights: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Sum finite blend features. Missing and non-finite values are left out.

    A missing similarity is not a measured zero, and an infinite mass error is
    not a real penalty. Both scorers call this so a NaN cannot change rank in
    one path and disappear in the other.
    """
    score = np.zeros(len(frame), dtype=np.float64)
    used = np.zeros(len(frame), dtype=bool)
    for key, coef in weights.items():
        if key not in frame.columns:
            continue
        values = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=np.float64)
        present = np.isfinite(values)
        score = np.where(present, score + float(coef) * np.where(present, values, 0.0), score)
        used = used | present
    return score, used


def apply_blend(frame: pd.DataFrame, weights: Mapping[str, float]) -> np.ndarray:
    """Apply the analytical blend. Generator-only rows use ``generator_score``."""
    score, used = linear_blend(frame, weights)
    if "generator_score" in frame.columns:
        generated = pd.to_numeric(frame["generator_score"], errors="coerce").to_numpy(dtype=np.float64)
        generated_ok = np.isfinite(generated) & ~used
        score = np.where(generated_ok, generated, score)
    return score.astype(np.float64)


def fuse_sources(
    sources: Sequence[SOURCE_FRAME],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Union the source tables and keep the higher blend for each structure."""
    parts: List[pd.DataFrame] = []
    for source, frame in sources:
        if frame is None or len(frame) == 0:
            continue
        required = {"molecule_id", "smiles", "inchikey14"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"source {source} is missing columns {sorted(missing)}")
        part = frame.copy()
        part["source"] = source
        parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=["molecule_id", "smiles", "inchikey14", "source", "blend_score", "n_sources"]
        )

    stacked = pd.concat(parts, ignore_index=True)
    stacked["blend_score"] = apply_blend(stacked, weights)
    stacked = stacked.sort_values(
        ["molecule_id", "blend_score"],
        ascending=[True, False],
        kind="mergesort",
    )
    counts = (
        stacked.groupby(["molecule_id", "inchikey14"], sort=False)["source"]
        .nunique()
        .rename("n_sources")
        .reset_index()
    )
    deduped = stacked.drop_duplicates(["molecule_id", "inchikey14"], keep="first")
    return deduped.merge(counts, on=["molecule_id", "inchikey14"], how="left")


def topk_smiles(fused: pd.DataFrame, k: int = 25) -> Dict[str, List[str]]:
    """Ranked SMILES per molecule, already deduplicated on inchikey14."""
    ranked: Dict[str, List[str]] = {}
    if fused.empty:
        return ranked
    ordered = fused.sort_values(
        ["molecule_id", "blend_score"],
        ascending=[True, False],
        kind="mergesort",
    )
    for molecule_id, group in ordered.groupby("molecule_id", sort=False):
        ranked[str(molecule_id)] = group["smiles"].astype(str).head(k).tolist()
    return ranked


def iter_source_names(sources: Iterable[SOURCE_FRAME]) -> List[str]:
    return [name for name, frame in sources if frame is not None and len(frame) > 0]

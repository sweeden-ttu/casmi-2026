"""P5 wiring run: one database source, scored with the Evidently blend.

IV4 adds a DualLGD frame when ``experiments/runs/DN6/candidates.json`` exists.
The run writes ``experiments/runs/P5/candidates.json`` for the IV3 gate.
It does not fit a new blend. The synthetic pool is not a leaderboard result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping
import json
import numpy as np
import pandas as pd

from casmi26.chemistry import score_mrr_at_25
from casmi26.features import clean_and_normalize_spectrum, smiles_to_fingerprint, spectrum_to_binned_vector
from casmi26.fusion import fuse_sources, topk_smiles
from casmi26.models import CASMIPipeline, SpectralLibraryIndex, SpectrumToFingerprintModel
from casmi26.retrieval import build_candidate_pool_from_sources
from experiments.tracker import REPO_ROOT, RUNS_ROOT


def _fit_fingerprint_model(train_df: pd.DataFrame) -> SpectrumToFingerprintModel:
    n_rows = len(train_df)
    binned = np.zeros((n_rows, 1536), dtype=np.float32)
    masses = np.zeros(n_rows, dtype=np.float32)
    positive = np.zeros(n_rows, dtype=np.float32)
    fingerprints = np.zeros((n_rows, 2215), dtype=np.float32)
    for index, row in enumerate(train_df.itertuples(index=False)):
        cleaned_mzs, cleaned_probs = clean_and_normalize_spectrum(
            row.ms2_mzs,
            row.ms2_normalized_intensities,
            row.precursor_mz,
        )
        binned[index] = spectrum_to_binned_vector(cleaned_mzs, cleaned_probs, row.precursor_mz)
        masses[index] = float(row.precursor_mz)
        positive[index] = 1.0 if str(row.ionization_mode).lower().startswith("pos") else 0.0
        fingerprints[index] = smiles_to_fingerprint(row.normalized_smiles)
    model = SpectrumToFingerprintModel(use_torch_if_available=False, ridge_alpha=5.0)
    model.fit(binned, masses, positive, fingerprints)
    return model


def _load_dn6_frame() -> pd.DataFrame:
    path = RUNS_ROOT / "DN6" / "candidates.json"
    if not path.is_file():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload)


def run_p5(weights: Mapping[str, float]) -> Dict[str, object]:
    """Score the synthetic validation molecules and fuse the available sources."""
    synth = REPO_ROOT / "data" / "synthetic"
    train_df = pd.read_parquet(synth / "train_synthetic.parquet")
    query_df = pd.read_parquet(synth / "test_synthetic.parquet")
    truth_df = pd.read_parquet(synth / "val_truth_synthetic.parquet")
    pool = build_candidate_pool_from_sources(
        train_df=train_df,
        candidate_parquet_paths=[
            synth / "coconut_candidates.parquet",
            synth / "pubchem_candidates.parquet",
        ],
    )
    pipeline = CASMIPipeline(
        candidate_pool=pool,
        spectral_index=SpectralLibraryIndex.build(train_df),
        fp_model=_fit_fingerprint_model(train_df),
        ppm_tolerance=15.0,
    )
    records: List[dict] = []
    truth_ids = set(truth_df["molecule_id"].astype(str))
    for molecule_id, group in query_df.groupby("molecule_id", sort=False):
        if str(molecule_id) not in truth_ids:
            continue
        features = pipeline.build_molecule_candidate_features(group)
        if features.empty:
            continue
        features = features.copy()
        features["molecule_id"] = str(molecule_id)
        features["blend_input_score"] = pipeline.score_candidates(features, weights=dict(weights))
        records.extend(features.to_dict(orient="records"))

    database = pd.DataFrame(records)
    sources = [("database", database)]
    dn6 = _load_dn6_frame()
    if not dn6.empty:
        sources.append(("dn6", dn6))
    fused = fuse_sources(sources, weights)
    ranked = topk_smiles(fused, k=25)
    ordered_truth = truth_df[truth_df["molecule_id"].astype(str).isin(list(ranked))].copy()
    ordered_truth["molecule_id"] = ordered_truth["molecule_id"].astype(str)
    predictions = [ranked[mid] for mid in ordered_truth["molecule_id"].tolist()]
    mrr, per_molecule = score_mrr_at_25(
        predictions,
        ordered_truth["inchikey14"].astype(str).tolist(),
        ground_truth_is_inchikey14=True,
    )
    truth_keys = dict(zip(ordered_truth["molecule_id"], ordered_truth["inchikey14"].astype(str)))
    in_pool = 0
    compact: List[dict] = []
    for molecule_id, truth_key in truth_keys.items():
        rows = database[database["molecule_id"] == molecule_id] if not database.empty else database
        present = bool((rows["inchikey14"].astype(str) == truth_key).any()) if len(rows) else False
        in_pool += int(present)
        if rows.empty:
            continue
        kept = rows.sort_values("blend_input_score", ascending=False).head(40)
        for row in kept.itertuples(index=False):
            compact.append(
                {
                    "molecule_id": molecule_id,
                    "smiles": str(row.smiles),
                    "inchikey14": str(row.inchikey14),
                    "blend_score": float(row.blend_input_score),
                    "source": "database",
                    "is_truth": str(row.inchikey14) == truth_key,
                }
            )
    out_dir = RUNS_ROOT / "P5"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates.json").write_text(json.dumps(compact), encoding="utf-8")
    n_molecules = int(len(ordered_truth))
    return {
        "status": "wiring_complete",
        "scope": "synthetic",
        "leaderboard_run": False,
        "n_molecules": n_molecules,
        "truth_in_pool": (in_pool / n_molecules) if n_molecules else 0.0,
        "mrr_at_25": mrr,
        "mrr_min": float(per_molecule.min()) if len(per_molecule) else 0.0,
        "sources": [name for name, _frame in sources if _frame is not None and len(_frame) > 0],
        "dn6_rows": int(len(dn6)),
        "weights_changed": False,
        "next_action": (
            "P5 fused the database list with the Evidently blend. "
            "Add experiments/runs/DN6/candidates.json and rerun so IV4 enters the merge. "
            "Do not submit this synthetic MRR."
        ),
    }

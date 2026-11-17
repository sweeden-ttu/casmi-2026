"""P5 merge keeps the higher Evidently blend and deduplicates on inchikey14."""

from __future__ import annotations

import numpy as np
import pandas as pd

from casmi26.fusion import apply_blend, fuse_sources
from casmi26.models import CASMIPipeline
from casmi26.weights import V001_ANALYTICAL


def test_blend_uses_the_evidently_coefficients():
    frame = pd.DataFrame(
        {
            "lib_max_entropy_sim": [1.0],
            "mass_error_ppm": [10.0],
        }
    )
    score = apply_blend(frame, V001_ANALYTICAL)
    assert abs(float(score[0]) - (4.5 - 0.8)) < 1e-9


def test_missing_features_rank_the_same_in_both_scorers():
    weights = {"lib_max_entropy_sim": 4.5, "mass_error_ppm": -0.08}
    frame = pd.DataFrame(
        {
            "lib_max_entropy_sim": [np.nan, 0.0, 1.0, np.nan],
            "mass_error_ppm": [10.0, 10.0, np.nan, np.inf],
        }
    )

    class _NoReranker:
        reranker = None

    scored = CASMIPipeline.score_candidates(_NoReranker(), frame, weights)
    fused = apply_blend(frame, weights)
    assert np.array_equal(scored, fused)
    assert scored[0] == scored[1]
    assert float(scored[3]) == 0.0


def test_generator_only_rows_keep_their_score():
    frame = pd.DataFrame({"generator_score": [0.25, 0.75]})
    score = apply_blend(frame, V001_ANALYTICAL)
    assert list(score) == [0.25, 0.75]


def test_fusion_dedupes_and_keeps_the_higher_blend():
    database = pd.DataFrame(
        {
            "molecule_id": ["m1", "m1"],
            "smiles": ["AAA", "BBB"],
            "inchikey14": ["KEYAAAA", "KEYBBBB"],
            "lib_max_entropy_sim": [0.2, 0.9],
        }
    )
    generated = pd.DataFrame(
        {
            "molecule_id": ["m1"],
            "smiles": ["AAA-from-generator"],
            "inchikey14": ["KEYAAAA"],
            "lib_max_entropy_sim": [0.95],
        }
    )
    fused = fuse_sources([("database", database), ("dn6", generated)], V001_ANALYTICAL)
    kept = fused.set_index("inchikey14")
    assert len(fused) == 2
    assert kept.loc["KEYAAAA", "source"] == "dn6"
    assert int(kept.loc["KEYAAAA", "n_sources"]) == 2
    assert kept.loc["KEYBBBB", "smiles"] == "BBB"

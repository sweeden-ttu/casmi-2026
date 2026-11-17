"""NPC2 keeps a ranker only when it beats the NPComposer generation order."""

from __future__ import annotations

import pandas as pd

from experiments.npc_run import train_to_beat_generations


def _frame(separated: bool) -> pd.DataFrame:
    rows = []
    for index in range(8):
        molecule_id = f"mol_{index:02d}"
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        truth = alphabet[index] * 14
        other = alphabet[index + 13] * 14
        for rank in range(1, 6):
            is_truth = rank == 5
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "inchikey14": truth if is_truth else other,
                    "generation_rank": rank,
                    "is_truth": is_truth,
                    "mass_error_ppm": 0.0 if is_truth and separated else 10.0,
                    "np_likeness_score": 1.0 if is_truth and separated else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_ranker_is_kept_only_when_it_beats_the_generated_order():
    improved = train_to_beat_generations(_frame(True), seed=3407)
    assert improved["adopted"] is True
    assert improved["learned_mrr_at_25"] > improved["baseline_mrr_at_25"]
    assert improved["weights_changed"] is False

    unchanged = train_to_beat_generations(_frame(False), seed=3407)
    assert unchanged["adopted"] is False
    assert unchanged["coefficients"] == {}

# CASMI experiment tracker

Updated 2026-11-17T14:29:48+00:00.

Resume here. `status.json` is the machine-readable copy of this page, and `log.jsonl` is the append-only history. Rerun with the kagg interpreter from the repository root:

```bash
/home/sweeden/.conda/envs/kagg/bin/python -m experiments
```

A finished checkpoint run is `python -m experiments <EXP_ID>` after the artifact in the next-action column is in place. Seed for every run is 3407. Active blend is `experiments/weights/current.json`. Do not retune that blend on `data/synthetic`; those SMILES are not molecules and their inchikey14 values are hashes.

Weights version: `v002`. Evidently dataset: `01a1e9e9-fd4f-742c-88c3-02721042396e`.

## Queue

| Order | Experiment | Status | Next action |
| --- | --- | --- | --- |
| 1 | GM0 | blocked | Train with the conditioner first, then the whole GP-MoLFormer, seed 3407. Place data/raw/train.parquet, artifacts/dreams/backbone.pt, and artifacts/GM0/gp-molformer, then rerun python -m experiments GM0. |
| 2 | DN6 | blocked | Export the official DualLGD checkpoint to artifacts/DN6/duallgd.pt, generate under the formula shortlist, and write experiments/runs/DN6/candidates.json. |
| 3 | IV4 | blocked | IV4 waits on DN6. P5 already merges a DualLGD frame when experiments/runs/DN6/candidates.json is present. |
| 4 | P5 | wiring_complete | P5 fused the database list with the Evidently blend. Add experiments/runs/DN6/candidates.json and rerun so IV4 enters the merge. Do not submit this synthetic MRR. |
| 5 | EN1 | blocked | Build the DreaMS cache with the self-supervised backbone and write artifacts/EN1/dreams_embeddings.parquet. |
| 6 | DN1 | blocked | Score DiffMS structures with the unchanged retrieval blend and write experiments/runs/DN1/candidates.json. |
| 7 | P4 | blocked | Train the projector and LoRA only after the base model is allowed. Keep database retrieval out of this run. |
| 8 | IV3 | not_adopted | Gate coefficients are stored inactive under iv3_gate. Refit on real spectra with the Tanimoto 0.7 absent rule before turning the gate on in P5. |
| 9 | EN4 | blocked | Cache MIST embeddings under the top-5 formula shortlist and fit the shared heads on that cache. |
| 10 | RR1 | blocked | Train the spectrum and molecule encoders with InfoNCE and same-formula negatives. Do not replace P2 until it beats RR2. |
| 11 | RR5 | blocked | LoRA-tune the SMILES encoder under the same InfoNCE loss. Keep it on the P3 path. |
| 12 | FH0 | blocked | Train the hierarchical GNN molecule encoder against the DreaMS embedding. P3 ships only if it beats P2. |
| 13 | IV1 | blocked | Concatenate a learned 128-d adduct vector to the DreaMS embedding and report MRR by adduct. |

## Environment

- Python: `/home/sweeden/.conda/envs/kagg/bin/python`
- CUDA: `False`
- RDKit: `False`
- Local spectra: `data/synthetic` only. `data/raw/train.parquet` is not on disk.

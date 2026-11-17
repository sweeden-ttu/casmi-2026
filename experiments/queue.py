"""Checkpoint paths for the experiments that cannot be trained on this machine.

A row is blocked until every listed path exists. The notes are the handoff:
the next agent drops the artifact at that path and reruns ``python -m experiments``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# (path relative to the repo, what the file must be)
Requirement = Tuple[str, str]

REQUIREMENTS: Dict[str, List[Requirement]] = {
    "GM0": [
        ("data/raw/train.parquet", "Competition train.parquet. data/synthetic is not a substitute."),
        ("artifacts/dreams/backbone.pt", "DreaMS self-supervised backbone, not a NIST fine-tune."),
        ("artifacts/GM0/gp-molformer", "ibm-research/GP-MoLFormer-Uniq snapshot."),
    ],
    "DN6": [
        ("artifacts/DN6/duallgd.pt", "DualLGD TorchScript export. The P5 flag reads this path."),
    ],
    "IV4": [
        ("experiments/runs/DN6/candidates.json", "Structures written by a finished DN6 run, one record per candidate."),
    ],
    "EN1": [
        ("artifacts/EN1/dreams_embeddings.parquet", "Frozen DreaMS embedding cache for train and validation spectra."),
    ],
    "DN1": [
        ("artifacts/DN1/diffms.pt", "Official DiffMS weights plus the MIST encoder they condition on."),
        ("artifacts/EN4/mist.pt", "MIST formula transformer. DiffMS is conditioned on it."),
    ],
    "P4": [
        ("artifacts/dreams/backbone.pt", "DreaMS backbone whose soft tokens feed the projector."),
        ("artifacts/P4/spectral.pt", "Projector checkpoint from train_spectral_llm stages 2 and 3."),
        ("artifacts/P4/lora", "LoRA adapter for the spectral language model."),
    ],
    "EN4": [
        ("artifacts/EN4/mist.pt", "MIST weights trained on MassSpecGym, with the authors' training code."),
    ],
    "RR1": [
        ("artifacts/EN1/dreams_embeddings.parquet", "Spectrum side of the InfoNCE head is the frozen DreaMS cache."),
    ],
    "RR5": [
        ("artifacts/RR5/chemberta", "PubChem-pretrained SMILES encoder whose training data has been checked."),
        ("artifacts/EN1/dreams_embeddings.parquet", "Spectrum side stays the custom encoder cache."),
    ],
    "FH0": [
        ("artifacts/EN1/dreams_embeddings.parquet", "InfoNCE target on the spectrum side is the DreaMS embedding."),
        ("data/hgraph", "Hierarchical graph cache. RDKit is also required to rebuild a missing cache."),
    ],
    "IV1": [
        ("artifacts/EN1/dreams_embeddings.parquet", "The 128-d adduct vector is concatenated to the 1024-d DreaMS embedding."),
    ],
}

GPU_REQUIRED = {"GM0", "DN6", "DN1", "P4", "RR1", "RR5", "FH0"}

NEXT_COMMAND = {
    "GM0": "Train with the conditioner first, then the whole GP-MoLFormer, seed 3407. data/raw/train.parquet and artifacts/GM0/gp-molformer are in place. Place artifacts/dreams/backbone.pt and rerun on a GPU: python -m experiments GM0.",
    "DN6": "Export the official DualLGD checkpoint to artifacts/DN6/duallgd.pt, generate under the formula shortlist, and write experiments/runs/DN6/candidates.json.",
    "IV4": "Rerun after DN6 has written candidates.json. P5 rescores those rows with the retrieval blend before the merge.",
    "EN1": "Build the DreaMS cache with the self-supervised backbone and write artifacts/EN1/dreams_embeddings.parquet.",
    "DN1": "Score DiffMS structures with the unchanged retrieval blend and write experiments/runs/DN1/candidates.json.",
    "P4": "Train the projector and LoRA only after the base model is allowed. Keep database retrieval out of this run.",
    "EN4": "Cache MIST embeddings under the top-5 formula shortlist and fit the shared heads on that cache.",
    "RR1": "Train the spectrum and molecule encoders with InfoNCE and same-formula negatives. Do not replace P2 until it beats RR2.",
    "RR5": "LoRA-tune the SMILES encoder under the same InfoNCE loss. Keep it on the P3 path.",
    "FH0": "Train the hierarchical GNN molecule encoder against the DreaMS embedding. P3 ships only if it beats P2.",
    "IV1": "Concatenate a learned 128-d adduct vector to the DreaMS embedding and report MRR by adduct.",
}

# [Colab Cell 1] Install pinned competition dependencies & configure environment
# Packages are installed by the prerequisites cell (rl77_env.ensure_packages): offline from the
# wheels bundled by notebook 2 when present, otherwise from PyPI with rdkit==2026.03.3 pinned.

import os
import math
import time
import random
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import inchi

RDLogger.DisableLog("rdApp.*")


class CFG:
    # Paths: competition parquet files and this notebook's output folder (/kaggle/working on Kaggle)
    DATA_DIR = ENV.competition_dir()
    OUTPUT_DIR = ENV.work_dir()
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
    # Optional pre-trained ChemFormer encoder state_dict (GLMR freezes a pre-trained encoder).
    MOL_ENCODER_CKPT = os.environ.get("RL77_MOL_ENCODER_CKPT", "")

    # Spectral preprocessing constants
    REL_INTENSITY_FLOOR = 0.01       # Drop peaks < 1% of base peak
    PRECURSOR_CLIP_OFFSET = 2.0      # Clip fragments > precursor_mz + 2.0 Da
    C13_MASS_DIFF = 1.003355         # Carbon-13 isotope spacing (Da)
    C13_TOLERANCE_DA = 0.015         # Deisotoping tolerance window
    MAX_PEAKS = 61                   # GLMR spectral encoder sequence length (60 fragments + 1 precursor)
    ELECTRON_MASS_CORR = 0.000548    # Positive-mode electron mass correction for train.parquet

    # GLMR & Multi-Spectrum Architecture Hyperparameters (Appendix B of GLMR)
    D_MODEL = 256
    N_HEADS = 8
    MOL_ENC_LAYERS = 4               # ChemFormer bidirectional encoder layers
    SPEC_ENC_LAYERS = 6              # Spectral Multi-Head Attention layers
    MOL_DEC_LAYERS = 4               # ChemFormer autoregressive decoder layers
    MAX_SMILES_LEN = 256             # Token sequence cap (99.9% of train SMILES are <= 207 chars)
    TOP_K_CANDIDATES = W.int("nb1.retrieval.top_k")          # Pre-retrieval candidate pool size K
    BEAM_SIZE = W.int("nb1.decode.beam_size")                 # Beam search width during generative stage
    NUM_FG_CLASSES = 60              # Mass2SMILES auxiliary functional group head

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 64
    LR = 1e-4
    WEIGHT_DECAY = 0.1
    TEMPERATURE = W("nb1.loss.infonce_temperature")          # Dual-path InfoNCE temperature tau
    EPOCHS_STAGE1 = 300              # Upper bound; the wall-clock budget below usually ends Stage 1 first
    EPOCHS_STAGE2 = 30               # Upper bound for Stage 2
    STAGE1_MINUTES = float(os.environ.get("RL77_STAGE1_MINUTES", 55))
    STAGE2_MINUTES = float(os.environ.get("RL77_STAGE2_MINUTES", 35))
    MASS_TOLERANCE_PPM = W("nb1.retrieval.mass_tolerance_ppm")  # Candidate library mass filter window
    MIN_TOLERANCE_DA = W("nb1.retrieval.min_tolerance_da")
    TOP_N_SUBMISSION = 25            # Required ranked candidates per molecule_id

    # Real-data run sizes (a Kaggle GPU session is capped at 12 hours)
    N_TRAIN_MOLECULES = int(os.environ.get("RL77_N_TRAIN_MOLECULES", 40000))
    N_VAL_MOLECULES = int(os.environ.get("RL77_N_VAL_MOLECULES", 1000))
    N_VAL_EVAL = int(os.environ.get("RL77_N_VAL_EVAL", 250))  # val molecules decoded with beam search
    MAX_SPECTRA_LOADED = 32          # spectra kept per inchikey14 when streaming train.parquet
    MAX_SPECTRA_TRAIN = 8            # spectra sampled per molecule per training step
    MAX_SPECTRA_INFER = 16           # test.parquet has at most 9 spectra per molecule_id
    NUM_WORKERS = min(4, os.cpu_count() or 1)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = CFG.SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()
CFG.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

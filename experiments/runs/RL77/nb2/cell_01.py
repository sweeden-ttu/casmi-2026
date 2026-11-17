# [Colab Companion Cell 1] Curate, canonicalize (RDKit 2026.03.3), deduplicate by InChIKey14, and sort by exact mass
import os
import json
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


class IndexCFG:
    RAW_TRAIN_PARQUET = Path("/content/drive/MyDrive/casmi2026/data/train.parquet")
    EXTERNAL_SMILES_TXT = Path("/content/drive/MyDrive/casmi2026/data/coconut_pubchem_natural_products.smi")
    EXPORT_DIR = Path("/content/drive/MyDrive/casmi2026/kaggle_dataset_upload")
    CATALOG_PARQUET = EXPORT_DIR / "candidate_catalog_sorted.parquet"
    EMBEDDINGS_MMAP = EXPORT_DIR / "chemformer_embs_fp16.mmap"
    FAISS_INDEX_PATH = EXPORT_DIR / "chemformer_ivfflat.index"
    FORMULA_INDEX_PATH = EXPORT_DIR / "formula_offsets.npz"

    MIN_MASS_DA = 100.0   # Cover test.parquet range (157 - 1159 Da) with margin
    MAX_MASS_DA = 1250.0
    D_MODEL = 256
    SHARD_BATCH_SIZE = 1024


IndexCFG.EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def standardize_compound(raw_smiles: str) -> Optional[Tuple[str, str, float, str]]:
    """
    Applies RDKit 2026.03.3 tautomer canonicalization and returns:
    (canonical_smiles, inchikey14, exact_monoisotopic_mass, molecular_formula)
    """
    if not raw_smiles or not isinstance(raw_smiles, str):
        return None
    mol = Chem.MolFromSmiles(raw_smiles.strip())
    if mol is None:
        return None
    try:
        # Keep largest covalent fragment if salt is present
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

        canon_mol = tautomer_enumerator.Canonicalize(mol)
        exact_mass = float(Descriptors.ExactMolWt(canon_mol))
        if not (IndexCFG.MIN_MASS_DA <= exact_mass <= IndexCFG.MAX_MASS_DA):
            return None

        ik = inchi.MolToInchiKey(canon_mol)
        if not ik or len(ik) < 14:
            return None
        ik14 = ik[:14]

        canon_smi = Chem.MolToSmiles(canon_mol, isomericSmiles=False, canonical=True)
        formula = rdMolDescriptors.CalcMolFormula(canon_mol)
        return canon_smi, ik14, exact_mass, formula
    except Exception:
        return None


def build_mass_sorted_catalog(
    train_parquet_path: Path = IndexCFG.RAW_TRAIN_PARQUET,
    external_smi_path: Optional[Path] = IndexCFG.EXTERNAL_SMILES_TXT,
    output_parquet_path: Path = IndexCFG.CATALOG_PARQUET,
) -> pd.DataFrame:
    """
    Merges in-library training structures with external PubChem/COCONUT candidates,
    deduplicates strictly by inchikey14, and sorts ascending by exact_mass.
    """
    seen_ik14: Dict[str, Tuple[str, float, str, int]] = {}

    # 1. Load unique structures from train.parquet (Tier 1: In-library priority)
    if train_parquet_path.exists():
        train_df = pq.read_table(
            train_parquet_path, columns=["normalized_smiles", "inchikey14"]
        ).to_pandas()
        unique_train_smi = train_df["normalized_smiles"].dropna().unique()
        for smi in unique_train_smi:
            res = standardize_compound(str(smi))
            if res is not None:
                canon_smi, ik14, mass, formula = res
                if ik14 not in seen_ik14:
                    seen_ik14[ik14] = (canon_smi, mass, formula, 1)  # source=1 (train_lib)

    # 2. Merge external COCONUT / PubChem natural product structures (Tier 2: Known DB)
    if external_smi_path is not None and external_smi_path.exists():
        with open(external_smi_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                res = standardize_compound(parts[0])
                if res is not None:
                    canon_smi, ik14, mass, formula = res
                    if ik14 not in seen_ik14:
                        seen_ik14[ik14] = (canon_smi, mass, formula, 2)  # source=2 (external)

    records = [
        {
            "inchikey14": ik14,
            "canonical_smiles": vals[0],
            "exact_mass": np.float32(vals[1]),
            "molecular_formula": vals[2],
            "source_tier": np.uint8(vals[3]),
        }
        for ik14, vals in seen_ik14.items()
    ]

    catalog_df = pd.DataFrame(records)
    # Sort strictly by exact_mass ascending for O(log N) binary-search windowing
    catalog_df = catalog_df.sort_values("exact_mass", ascending=True).reset_index(drop=True)
    catalog_df["catalog_id"] = np.arange(len(catalog_df), dtype=np.int32)

    catalog_df.to_parquet(output_parquet_path, index=False, compression="zstd")
    print(f"Saved {len(catalog_df):,} unique InChIKey14 compounds to {output_parquet_path}")
    return catalog_df
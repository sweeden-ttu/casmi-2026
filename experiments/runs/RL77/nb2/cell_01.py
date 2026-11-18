# [Colab Companion Cell 1] Curate, canonicalize (RDKit 2026.03.3), deduplicate by InChIKey14, and sort by exact mass
import os
import json
import math
import multiprocessing as mp
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
    RAW_TRAIN_PARQUET = ENV.competition_dir() / "train.parquet"
    # External natural-product structures: COCONUT export shipped as the casmi26-rl77-candidates dataset.
    EXTERNAL_STRUCTURES = ENV.dataset_dir(ENV.CANDIDATES_SLUG, "coconut_structures.parquet", required=False)
    EXPORT_DIR = ENV.work_dir()          # published as the casmi26-rl77-nb2-store-outputs dataset
    CATALOG_PARQUET = EXPORT_DIR / "candidate_catalog_sorted.parquet"
    EMBEDDINGS_MMAP = EXPORT_DIR / "chemformer_embs_fp16.mmap"
    FAISS_INDEX_PATH = EXPORT_DIR / "chemformer_ivfflat.index"
    FORMULA_INDEX_PATH = EXPORT_DIR / "formula_offsets.npz"

    MIN_MASS_DA = 100.0   # Cover test.parquet range (157 - 1159 Da) with margin
    MAX_MASS_DA = 1250.0
    D_MODEL = 256
    SHARD_BATCH_SIZE = 1024
    N_PROCS = max(1, os.cpu_count() or 1)
    STANDARDIZE_MINUTES = float(os.environ.get("RL77_STANDARDIZE_MINUTES", 70))


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


def standardize_many(smiles: List[str], deadline: Optional["ENV.Deadline"] = None) -> List[Optional[Tuple[str, str, float, str]]]:
    """Parallel standardize_compound over a process pool; stops submitting work at the deadline."""
    results: List[Optional[Tuple[str, str, float, str]]] = []
    chunk = 20_000
    with mp.get_context("fork").Pool(IndexCFG.N_PROCS) as pool:
        for start in range(0, len(smiles), chunk):
            if deadline is not None and deadline.expired():
                ENV.log(f"Standardization budget reached after {start:,} of {len(smiles):,} structures")
                break
            results.extend(pool.map(standardize_compound, smiles[start:start + chunk], chunksize=256))
    return results


def load_precomputed_standardization(path: Path) -> Optional[List[Tuple[str, str, float, str]]]:
    """Tier-2 results from tools/standardize_candidates.py (this cell's standardize_compound run on a local CPU
    pool), used only when they were produced with the RDKit version and mass range in this session."""
    import rdkit

    if not path.exists():
        return None
    meta = json.loads((pq.read_schema(path).metadata or {}).get(b"rl77", b"{}"))
    expected = {"rdkit": rdkit.__version__, "min_mass_da": IndexCFG.MIN_MASS_DA, "max_mass_da": IndexCFG.MAX_MASS_DA}
    if any(meta.get(k) != v for k, v in expected.items()):
        ENV.log(f"Ignoring {path.name}: built with {meta}, need {expected}")
        return None
    df = pd.read_parquet(path, columns=["canonical_smiles", "inchikey14", "exact_mass", "molecular_formula"])
    df = df.dropna(subset=["inchikey14"])
    ENV.log(f"Using {path.name}: {len(df):,} standardized structures (RDKit {meta['rdkit']}, {meta.get('minutes')} min locally)")
    return list(df.itertuples(index=False, name=None))


def build_mass_sorted_catalog(
    train_parquet_path: Path = IndexCFG.RAW_TRAIN_PARQUET,
    external_path: Optional[Path] = IndexCFG.EXTERNAL_STRUCTURES,
    output_parquet_path: Path = IndexCFG.CATALOG_PARQUET,
) -> pd.DataFrame:
    """
    Merges in-library training structures with external PubChem/COCONUT candidates,
    deduplicates strictly by inchikey14, and sorts ascending by exact_mass.
    """
    seen_ik14: Dict[str, Tuple[str, float, str, int]] = {}
    deadline = ENV.Deadline(IndexCFG.STANDARDIZE_MINUTES)

    # 1. Load unique structures from train.parquet (Tier 1: In-library priority)
    if train_parquet_path.exists():
        train_smi = pq.read_table(train_parquet_path, columns=["normalized_smiles"])["normalized_smiles"]
        unique_train_smi = [s for s in pd.unique(train_smi.to_pandas().dropna()) if isinstance(s, str)]
        for res in standardize_many(unique_train_smi):  # tier 1 always completes
            if res is not None:
                canon_smi, ik14, mass, formula = res
                if ik14 not in seen_ik14:
                    seen_ik14[ik14] = (canon_smi, mass, formula, 1)  # source=1 (train_lib)
        ENV.log(f"Tier 1 (train.parquet): {len(seen_ik14):,} unique inchikey14")

    # 2. Merge external COCONUT / PubChem natural product structures (Tier 2: Known DB)
    if external_path is not None and Path(external_path).exists():
        external_path = Path(external_path)
        if external_path.is_dir():
            external_path = external_path / "coconut_structures.parquet"
        precomputed = load_precomputed_standardization(external_path.with_name("coconut_standardized.parquet"))
        if precomputed is not None:
            ext_results, ext_name = precomputed, "coconut_standardized.parquet (precomputed)"
        else:
            if external_path.suffix == ".parquet":
                ext_smi = pd.read_parquet(external_path, columns=["smiles"])["smiles"].dropna().tolist()
            else:
                ext_smi = [line.split()[0] for line in external_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            ext_results, ext_name = standardize_many(ext_smi, deadline=deadline), external_path.name
        n_before = len(seen_ik14)
        for res in ext_results:
            if res is not None:
                canon_smi, ik14, mass, formula = res
                if ik14 not in seen_ik14:
                    seen_ik14[ik14] = (canon_smi, mass, formula, 2)  # source=2 (external)
        ENV.log(f"Tier 2 ({ext_name}): +{len(seen_ik14) - n_before:,} new inchikey14")

    if not seen_ik14:
        raise RuntimeError("No candidate structures were standardized; check the train.parquet / COCONUT inputs.")

    records = [
        {
            "inchikey14": ik14,
            "canonical_smiles": vals[0],
            "exact_mass": float(vals[1]),  # float64: float32 rounds ~0.06 ppm at 1000 Da
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

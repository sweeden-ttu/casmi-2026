"""Precompute notebook 2's tier-2 standardization (COCONUT) on a local CPU pool.

Notebook 2 canonicalizes tautomers with RDKit 2026.03.3 for every catalog structure. On Kaggle's 4 CPUs the
494,774 COCONUT structures take ~4.5 h, so the 70-minute budget covered only 80,000 of them (nb2 v1). This
tool runs the *same* `standardize_compound` (extracted from nb2/cell_01.py, not a copy) over the COCONUT
export and writes `coconut_standardized.parquet` next to it; `build_notebooks.py` ships it in the
`casmi26-rl77-candidates` dataset and notebook 2 uses it when its RDKit version and mass range match.

  uv pip install --python .venv/bin/python "rdkit==2026.3.3"
  .venv/bin/python experiments/runs/RL77/tools/standardize_candidates.py [--procs 10]

CPU only (no GPU / neural work), so it is allowed locally under AGENTS.md.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RL77 = Path(__file__).resolve().parents[1]
REPO = RL77.parents[2]
CELL = RL77 / "nb2" / "cell_01.py"
SOURCE = REPO / "data" / "candidates" / "coconut" / "coconut_structures.parquet"
TARGET = SOURCE.with_name("coconut_standardized.parquet")


def load_standardizer() -> Tuple[object, str, float, float]:
    """Exec nb2's `standardize_compound` and IndexCFG mass limits from the cell source."""
    tree = ast.parse(CELL.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "standardize_compound")
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IndexCFG")
    limits = {
        t.targets[0].id: ast.literal_eval(t.value)
        for t in cfg.body
        if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name) and t.targets[0].id in ("MIN_MASS_DA", "MAX_MASS_DA")
    }
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    namespace = {
        "Chem": Chem, "Descriptors": Descriptors, "rdMolDescriptors": rdMolDescriptors, "inchi": inchi,
        "Optional": Optional, "Tuple": Tuple,
        "tautomer_enumerator": rdMolStandardize.TautomerEnumerator(),
        "IndexCFG": type("IndexCFG", (), limits),
    }
    src = ast.get_source_segment(CELL.read_text(encoding="utf-8"), fn)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(CELL), "exec"), namespace)
    return namespace["standardize_compound"], src, limits["MIN_MASS_DA"], limits["MAX_MASS_DA"]


_FN = None


def _init() -> None:
    global _FN
    _FN = load_standardizer()[0]


def _run(smiles: str):
    return _FN(smiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--procs", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()

    import rdkit

    _, fn_src, lo, hi = load_standardizer()
    smiles = pd.read_parquet(args.source, columns=["smiles"])["smiles"].tolist()
    t0, out = time.time(), []
    with mp.get_context("fork").Pool(args.procs, initializer=_init) as pool:
        for i, res in enumerate(pool.imap(_run, smiles, chunksize=256), 1):
            out.append(res)
            if i % 25_000 == 0:
                rate = i / (time.time() - t0)
                print(f"{i:,}/{len(smiles):,} ({rate:,.0f}/s, ~{(len(smiles) - i) / rate / 60:.0f} min left)", flush=True)

    frame = pd.DataFrame({
        "smiles": smiles,
        "canonical_smiles": [r[0] if r else None for r in out],
        "inchikey14": [r[1] if r else None for r in out],
        "exact_mass": [r[2] if r else None for r in out],
        "molecular_formula": [r[3] if r else None for r in out],
    })
    meta = {
        "rdkit": rdkit.__version__, "min_mass_da": lo, "max_mass_da": hi,
        "standardizer_sha256": hashlib.sha256(fn_src.encode()).hexdigest(),
        "source": args.source.name, "source_rows": len(smiles), "standardized": int(frame["inchikey14"].notna().sum()),
        "minutes": round((time.time() - t0) / 60, 1), "procs": args.procs,
    }
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"rl77": json.dumps(meta).encode()})
    pq.write_table(table, args.target, compression="zstd")
    print(f"Wrote {args.target} ({meta['standardized']:,} of {len(smiles):,} in range) {json.dumps(meta)}")


if __name__ == "__main__":
    main()

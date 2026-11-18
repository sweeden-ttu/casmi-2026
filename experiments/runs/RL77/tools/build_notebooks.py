"""Assemble the RL77 notebooks, the prerequisites Kaggle dataset, and the kernel folders.

Inputs (all under experiments/runs/RL77):
  nbN/cell_NN.py        reviewed Python cells (one per notebook cell, in order)
  markdown/nbN.md       text cells from the rubric PDF, split by <!-- intro -->, <!-- before:cell_NN -->, <!-- end -->
  lib/rl77_env.py       runtime helpers (paths, packages, budgets)
  lib/rl77_weights.py   weights loader
  weights/current.json  latest Evidently weights snapshot (tools/evidently_weights.py pull)

Outputs:
  notebooks/<slug>.ipynb                         assembled notebooks (also copied into the kernel folders)
  build/datasets/casmi26-rl77-prereqs/           helper modules + rl77_weights.json + one module per notebook
  build/datasets/casmi26-rl77-candidates/        COCONUT structures for notebook 2
  build/kernels/<slug>/                          notebook + kernel-metadata.json for `kaggle kernels push`

Usage:
  .venv/bin/python experiments/runs/RL77/tools/build_notebooks.py [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RL77 = Path(__file__).resolve().parents[1]
REPO = RL77.parents[2]
BUILD = RL77 / "build"
OWNER = "scottweeden"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
PREREQS_SLUG = "casmi26-rl77-prereqs"
CANDIDATES_SLUG = "casmi26-rl77-candidates"
COCONUT_PARQUET = REPO / "data" / "candidates" / "coconut" / "coconut_structures.parquet"

NOTEBOOKS: Dict[str, Dict] = {
    "nb1": {
        "slug": "casmi26-rl77-nb1-glmr",
        "title": "CASMI26 RL77 NB1 GLMR",
        "module": "rl77_nb1_glmr",
        "output_slug": "casmi26-rl77-nb1-glmr-outputs",
        "gpu": True,
        "upstream": [],
        "candidates": False,
        "provides": "Two-stage GLMR checkpoint, retrieval library, validation split, submission",
    },
    "nb2": {
        "slug": "casmi26-rl77-nb2-offline-store",
        "title": "CASMI26 RL77 NB2 Offline Store",
        "module": "rl77_nb2_store",
        "output_slug": "casmi26-rl77-nb2-store-outputs",
        "gpu": False,
        "upstream": ["nb1"],
        "candidates": True,
        "provides": "Mass-sorted candidate catalog, FP16 ChemFormer memmap, FAISS index, formula index, offline wheels",
    },
    "nb3": {
        "slug": "casmi26-rl77-nb3-ensemble",
        "title": "CASMI26 RL77 NB3 Ensemble",
        "module": "rl77_nb3_ensemble",
        "output_slug": "casmi26-rl77-nb3-ensemble-outputs",
        "gpu": False,
        "upstream": ["nb1", "nb2"],
        "candidates": False,
        "provides": "Siamese ECFP4 and Mass2SMILES heads, fitted blend weights, ensemble submission",
    },
    "nb4": {
        "slug": "casmi26-rl77-nb4-formula-filter",
        "title": "CASMI26 RL77 NB4 Formula Filter",
        "module": "rl77_nb4_formula",
        "output_slug": "casmi26-rl77-nb4-formula-outputs",
        "gpu": False,
        "upstream": ["nb1", "nb2", "nb3"],
        "candidates": False,
        "provides": "Formula priors for validation/test, fitted formula score weights, formula-constrained submission",
    },
    "nb5": {
        "slug": "casmi26-rl77-nb5-qrl",
        "title": "CASMI26 RL77 NB5 QRL",
        "module": "rl77_nb5_qrl",
        "output_slug": "casmi26-rl77-nb5-qrl-outputs",
        "gpu": True,
        "upstream": ["nb1", "nb2", "nb3", "nb4"],
        "candidates": False,
        "provides": "Stage 2.5 Q-learning decoder, Q-guided beam metrics, final submission",
    },
}


def kaggle_slug(title: str) -> str:
    """The slug Kaggle gives a *new* kernel: it is derived from the title, not from kernel-metadata `id`."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


for _nb, _spec in NOTEBOOKS.items():
    assert kaggle_slug(_spec["title"]) == _spec["slug"], (
        f"{_nb}: title {_spec['title']!r} becomes slug {kaggle_slug(_spec['title'])!r} on Kaggle, not {_spec['slug']!r}"
    )
    assert _spec["output_slug"] not in {v["slug"] for v in NOTEBOOKS.values()}, f"{_nb}: output slug collides with a kernel"

MARKER = re.compile(r"^<!--\s*(intro|end|before:(cell_\d+))\s*-->\s*$", re.M)


def cell_files(nb: str) -> List[Path]:
    return sorted((RL77 / nb).glob("cell_*.py"))


def read_markdown(nb: str) -> Dict[str, str]:
    text = (RL77 / "markdown" / f"{nb}.md").read_text(encoding="utf-8")
    parts: Dict[str, str] = {}
    matches = list(MARKER.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(2) or m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts[key] = text[m.end():end].strip()
    return parts


def weights_payload() -> Dict:
    return json.loads((RL77 / "weights" / "current.json").read_text(encoding="utf-8"))


def module_header(nb: str) -> str:
    lines = [
        f"# Auto-generated by tools/build_notebooks.py from experiments/runs/RL77/{nb}/cell_*.py. Do not edit.",
        "import rl77_env as ENV",
        "from rl77_weights import load_rl77_weights",
        "W = load_rl77_weights()",
    ]
    for up in NOTEBOOKS[nb]["upstream"]:
        lines.append(f"from {NOTEBOOKS[up]['module']} import *  # noqa: F401,F403")
    return "\n".join(lines) + "\n\n"


def module_source(nb: str) -> str:
    chunks = [module_header(nb)]
    for path in cell_files(nb):
        chunks.append(f"# ---- {nb}/{path.name} " + "-" * 40 + "\n" + path.read_text(encoding="utf-8").rstrip() + "\n\n")
    return "".join(chunks)


def prereq_code(nb: str) -> str:
    imports = "\n".join(
        f"from {NOTEBOOKS[up]['module']} import *  # {up}: {NOTEBOOKS[up]['title']}" for up in NOTEBOOKS[nb]["upstream"]
    )
    return f'''# Prerequisites: locate the RL77 helper modules, install pinned packages, load the Evidently weights snapshot
import glob, os, sys
from pathlib import Path

_lib_hits = sorted(glob.glob("/kaggle/input/**/rl77_env.py", recursive=True))
if _lib_hits:
    RL77_LIB = Path(_lib_hits[0]).parent
elif os.environ.get("RL77_LIB_DIR"):
    RL77_LIB = Path(os.environ["RL77_LIB_DIR"])  # e.g. experiments/runs/RL77/build/datasets/{PREREQS_SLUG}
else:
    raise FileNotFoundError("Attach the {OWNER}/{PREREQS_SLUG} dataset or set RL77_LIB_DIR.")
sys.path.insert(0, str(RL77_LIB))

import rl77_env as ENV
ENV.ensure_packages()                      # rdkit==2026.03.3 + faiss-cpu (offline wheels when attached)
from rl77_weights import load_rl77_weights
W = load_rl77_weights()                    # every blend weight / reward coefficient in this notebook reads W("...")
{imports}
ENV.log(f"Competition data: {{ENV.competition_dir()}} | outputs: {{ENV.work_dir()}}")
'''.rstrip() + "\n"


def prereq_markdown(nb: str, payload: Dict) -> str:
    spec = NOTEBOOKS[nb]
    rows = [
        f"| Competition data | `{COMPETITION}` | `train.parquet`, `test.parquet`, `sample_submission.csv` |",
        f"| RL77 prerequisites | `{OWNER}/{PREREQS_SLUG}` | `rl77_env.py`, `rl77_weights.py`, `rl77_weights.json` (Evidently snapshot `{payload['version']}`), upstream modules |",
    ]
    if spec["candidates"]:
        rows.append(f"| Candidate structures | `{OWNER}/{CANDIDATES_SLUG}` | `coconut_structures.parquet` (COCONUT natural products) |")
    for up in spec["upstream"]:
        rows.append(f"| Notebook {up[-1]} outputs | `{OWNER}/{NOTEBOOKS[up]['output_slug']}` | {NOTEBOOKS[up]['provides']} |")
    upstream_txt = (
        "Run notebooks " + ", ".join(u[-1] for u in spec["upstream"]) + " first and publish their outputs "
        "(`tools/kaggle_pipeline.py publish-output`)." if spec["upstream"] else "This is the first notebook in the chain."
    )
    ds = payload.get("evidently", {}).get("dataset_id") or "not yet published"
    return "\n".join([
        "## Prerequisites",
        "",
        "| Input | Kaggle source | Provides |",
        "| --- | --- | --- |",
        *rows,
        "",
        f"- **Order:** {upstream_txt}",
        f"- **Accelerator:** {'GPU (T4)' if spec['gpu'] else 'CPU (no GPU quota used)'}; internet on for the test run so `ensure_packages()` can reach PyPI. Offline runs install from the wheels in notebook 2's output dataset.",
        "- **Packages:** `rdkit==2026.03.3` (the metric's canonicalizer) and `faiss-cpu` are installed if the image lacks them; `torch`, `pyarrow`, `pandas`, `scipy` come from the Kaggle image.",
        f"- **Weights:** all blend weights, reward coefficients, and tolerances come from Evidently dataset `casmi-rl77-weights` (version `{payload['version']}`, id `{ds}`) through `W(\"...\")`.",
        f"- **Outputs:** everything written to `/kaggle/working` is published as `{OWNER}/{spec['output_slug']}` for the next notebook.",
    ])


def build_notebook(nb: str, payload: Dict) -> nbformat.NotebookNode:
    md = read_markdown(nb)
    cells = [new_markdown_cell(md["intro"]), new_markdown_cell(prereq_markdown(nb, payload)), new_code_cell(prereq_code(nb))]
    for path in cell_files(nb):
        before = md.get(path.stem)
        if before:
            cells.append(new_markdown_cell(before))
        cells.append(new_code_cell(path.read_text(encoding="utf-8").rstrip()))
    if md.get("end"):
        cells.append(new_markdown_cell(md["end"]))
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return notebook


def kernel_metadata(nb: str, offline: bool = False) -> Dict:
    """Kaggle kernel metadata; `offline` turns internet off, as the competition requires for a scored version."""
    spec = NOTEBOOKS[nb]
    sources = [f"{OWNER}/{PREREQS_SLUG}"]
    if spec["candidates"]:
        sources.append(f"{OWNER}/{CANDIDATES_SLUG}")
    sources += [f"{OWNER}/{NOTEBOOKS[up]['output_slug']}" for up in spec["upstream"]]
    meta = {
        "id": f"{OWNER}/{spec['slug']}",
        "title": spec["title"],
        "code_file": f"{spec['slug']}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": spec["gpu"],
        "enable_tpu": False,
        "enable_internet": not offline,
        "dataset_sources": sources,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
    }
    if spec["gpu"]:
        meta["machine_shape"] = "NvidiaTeslaT4"
    return meta


def build_prereqs(payload: Dict) -> Path:
    out = BUILD / "datasets" / PREREQS_SLUG
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(RL77 / "lib" / "rl77_env.py", out / "rl77_env.py")
    shutil.copy2(RL77 / "lib" / "rl77_weights.py", out / "rl77_weights.py")
    (out / "rl77_weights.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for nb, spec in NOTEBOOKS.items():
        (out / f"{spec['module']}.py").write_text(module_source(nb), encoding="utf-8")
    (out / "README.md").write_text(
        "# CASMI26 RL77 prerequisites\n\n"
        "Helper modules for the RL77 notebooks (experiments/runs/RL77 in the CASMI repository).\n\n"
        f"- `rl77_env.py`: locate inputs, install pinned packages, time budgets.\n"
        f"- `rl77_weights.py` + `rl77_weights.json`: Evidently snapshot `{payload['version']}` of the RL77 weights.\n"
        "- `rl77_nbN_*.py`: each notebook's cells as an importable module (drivers run only under `__main__`).\n",
        encoding="utf-8",
    )
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "CASMI26 RL77 prereqs",
        "id": f"{OWNER}/{PREREQS_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n", encoding="utf-8")
    return out


def build_candidates() -> Path:
    out = BUILD / "datasets" / CANDIDATES_SLUG
    out.mkdir(parents=True, exist_ok=True)
    target = out / "coconut_structures.parquet"
    if not target.exists() or target.stat().st_size != COCONUT_PARQUET.stat().st_size:
        shutil.copy2(COCONUT_PARQUET, target)
    # Optional: tools/standardize_candidates.py output (notebook 2 skips its in-kernel tier-2 standardization)
    standardized = COCONUT_PARQUET.with_name("coconut_standardized.parquet")
    if standardized.exists():
        shutil.copy2(standardized, out / standardized.name)
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "CASMI26 RL77 candidates",
        "id": f"{OWNER}/{CANDIDATES_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n", encoding="utf-8")
    return out


def check(nb: str) -> int:
    """pyflakes over prerequisites + upstream cells (inlined instead of star imports) + this notebook."""
    pieces = ["import rl77_env as ENV\nfrom rl77_weights import load_rl77_weights\nW = load_rl77_weights()\n"]
    for up in NOTEBOOKS[nb]["upstream"]:
        pieces += [p.read_text(encoding="utf-8") for p in cell_files(up)]
    pieces += [p.read_text(encoding="utf-8") for p in cell_files(nb)]
    src = BUILD / "check" / f"{nb}_check.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("\n\n".join(pieces), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "pyflakes", str(src)], capture_output=True, text=True)
    noise = ("redefinition of unused", "imported but unused", "is assigned to but never used", "f-string is missing placeholders")
    problems = [line for line in result.stdout.splitlines() if not any(n in line for n in noise)]
    problems += [line for line in result.stderr.splitlines() if line.strip()]
    print(f"[check] {nb}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    for line in problems:
        print("   ", line.replace(str(src), f"{nb}_check.py"))
    return len(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="run pyflakes over each assembled notebook")
    parser.add_argument("--only", nargs="*", default=list(NOTEBOOKS), help="notebooks to assemble (default all)")
    parser.add_argument("--offline", nargs="*", default=[], choices=list(NOTEBOOKS),
                        help="build these kernels with internet off (required for a leaderboard submission); "
                             "packages then come from the wheels in casmi26-rl77-nb2-store-outputs")
    args = parser.parse_args()

    payload = weights_payload()
    prereqs = build_prereqs(payload)
    candidates = build_candidates()
    nb_dir = RL77 / "notebooks"
    nb_dir.mkdir(exist_ok=True)
    failures = 0
    for nb in args.only:
        if not cell_files(nb):
            print(f"[skip] {nb}: no cells")
            continue
        notebook = build_notebook(nb, payload)
        spec = NOTEBOOKS[nb]
        nb_path = nb_dir / f"{spec['slug']}.ipynb"
        nbformat.write(notebook, nb_path)
        kdir = BUILD / "kernels" / spec["slug"]
        kdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nb_path, kdir / nb_path.name)
        (kdir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(nb, offline=nb in args.offline), indent=2) + "\n", encoding="utf-8")
        print(f"[build] {nb}: {len(notebook.cells)} cells -> {nb_path.relative_to(REPO)}" + (" (internet off)" if nb in args.offline else ""))
        if args.check:
            failures += check(nb)
    print(f"[build] prereqs dataset: {prereqs.relative_to(REPO)} (weights {payload['version']})")
    print(f"[build] candidates dataset: {candidates.relative_to(REPO)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

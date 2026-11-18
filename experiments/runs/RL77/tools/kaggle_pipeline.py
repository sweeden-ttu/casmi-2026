"""Push, run, collect, and chain the RL77 Kaggle notebooks.

Each notebook writes its results to /kaggle/working. After a run, `fetch` downloads that folder and
the log, and `publish-output` turns it into a private Kaggle dataset that the next notebook attaches
(notebook 1 -> casmi26-rl77-nb1-glmr-outputs, and so on). Commands, from the repository root:

  PY=.venv/bin/python; T=experiments/runs/RL77/tools
  $PY $T/build_notebooks.py --check          # assemble notebooks + prerequisites dataset
  $PY $T/kaggle_pipeline.py push-datasets    # create/version casmi26-rl77-prereqs and casmi26-rl77-candidates
  $PY $T/kaggle_pipeline.py push nb1         # start a full Kaggle run of notebook 1
  $PY $T/kaggle_pipeline.py wait nb1         # poll until complete / error (prints the log tail on error)
  $PY $T/kaggle_pipeline.py fetch nb1        # download /kaggle/working + log to build/outputs/nb1
  $PY $T/evidently_weights.py adopt experiments/runs/RL77/build/outputs/nb1   # fold fitted weights/metrics into Evidently
  $PY $T/kaggle_pipeline.py publish-output nb1  # outputs -> dataset for notebooks 2-5
  $PY $T/kaggle_pipeline.py submit nb5 --yes    # leaderboard submission (uses a daily submission slot)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import BUILD, CANDIDATES_SLUG, COMPETITION, NOTEBOOKS, OWNER, PREREQS_SLUG, REPO  # noqa: E402

KAGGLE = str(REPO / ".venv" / "bin" / "kaggle")
OUTPUTS = BUILD / "outputs"


def kaggle(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = [KAGGLE, *args]
    print("$ kaggle " + " ".join(args), flush=True)
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if capture and result.stdout.strip():
        print(result.stdout.strip()[-4000:])
    if capture and result.stderr.strip():
        print(result.stderr.strip()[-4000:], file=sys.stderr)
    if check and result.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed ({result.returncode})")
    return result


def dataset_exists(ref: str) -> bool:
    return kaggle("datasets", "status", ref, check=False).returncode == 0


def upsert_dataset(folder: Path, message: str) -> None:
    meta = json.loads((folder / "dataset-metadata.json").read_text())
    kernel_slugs = {spec["slug"] for spec in NOTEBOOKS.values()}
    if meta["id"].split("/", 1)[1] in kernel_slugs:  # kernels and datasets share one URL namespace per user
        raise SystemExit(f"{meta['id']} collides with a kernel slug; Kaggle rejects it after the upload")
    if dataset_exists(meta["id"]):
        result = kaggle("datasets", "version", "-p", str(folder), "-m", message, "--dir-mode", "zip", "-q")
    else:
        result = kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "zip", "-q")
    # The CLI exits 0 even when the server rejects the dataset ("Dataset creation error: ...").
    if "error" in (result.stdout + result.stderr).lower():
        raise SystemExit(f"Kaggle rejected {meta['id']}: see the message above")
    wait_dataset_ready(meta["id"])


def wait_dataset_ready(ref: str, timeout_min: float = 30) -> None:
    t_end = time.time() + 60 * timeout_min
    while time.time() < t_end:
        out = kaggle("datasets", "status", ref, check=False).stdout.strip().lower()
        if "ready" in out:
            return
        if "error" in out:
            raise SystemExit(f"Dataset {ref} failed processing: {out}")
        time.sleep(20)
    raise SystemExit(f"Dataset {ref} not ready after {timeout_min} min")


def kernel_ref(nb: str) -> str:
    return f"{OWNER}/{NOTEBOOKS[nb]['slug']}"


def status(nb: str) -> str:
    out = kaggle("kernels", "status", kernel_ref(nb), check=False).stdout.lower()
    for state in ("complete", "error", "cancelacknowledged", "cancelrequested", "running", "queued"):
        if state in out.replace("_", "").replace(" ", ""):
            return state
    return out.strip() or "unknown"


def cmd_push_datasets(args) -> None:
    upsert_dataset(BUILD / "datasets" / PREREQS_SLUG, args.message)
    if args.with_candidates or not dataset_exists(f"{OWNER}/{CANDIDATES_SLUG}"):
        upsert_dataset(BUILD / "datasets" / CANDIDATES_SLUG, args.message)


def cmd_push(args) -> None:
    for up in NOTEBOOKS[args.nb]["upstream"]:
        ref = f"{OWNER}/{NOTEBOOKS[up]['output_slug']}"
        if not dataset_exists(ref):
            raise SystemExit(f"{args.nb} needs {ref}; run `fetch {up}` and `publish-output {up}` first.")
    kaggle("kernels", "push", "-p", str(BUILD / "kernels" / NOTEBOOKS[args.nb]["slug"]))


def cmd_wait(args) -> None:
    t_end = time.time() + 3600 * args.hours
    last = None
    while time.time() < t_end:
        state = status(args.nb)
        if state != last:
            print(f"{time.strftime('%H:%M:%S')} {args.nb}: {state}", flush=True)
            last = state
        if state in ("complete", "error", "cancelacknowledged"):
            if state != "complete":
                cmd_fetch(args)
            raise SystemExit(0 if state == "complete" else 2)
        time.sleep(args.poll)
    raise SystemExit(f"{args.nb} still {last} after {args.hours} h")


def cmd_fetch(args) -> None:
    dest = OUTPUTS / args.nb
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    kaggle("kernels", "output", kernel_ref(args.nb), "-p", str(dest), "-o", check=False)
    logs = sorted(dest.glob("*.log"))
    for log in logs:
        tail = log.read_text(errors="replace")[-6000:]
        print(f"---- {log.name} (tail) ----\n{tail}")
    print(f"Fetched {sum(1 for _ in dest.rglob('*') if _.is_file())} files into {dest}")


def cmd_publish_output(args) -> None:
    src = OUTPUTS / args.nb
    if not (src / "run_manifest.json").exists():
        raise SystemExit(f"{src} has no run_manifest.json; the run did not finish. Fix and rerun before publishing.")
    folder = BUILD / "datasets" / NOTEBOOKS[args.nb]["output_slug"]
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(src, folder, ignore=shutil.ignore_patterns("*.log", "__results__*", "*.ipynb", "__output__*"))
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": NOTEBOOKS[args.nb]["title"] + " outputs"[: max(0, 50 - len(NOTEBOOKS[args.nb]["title"]))],
        "id": f"{OWNER}/{NOTEBOOKS[args.nb]['output_slug']}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n")
    upsert_dataset(folder, args.message)


def cmd_submit(args) -> None:
    if not args.yes:
        raise SystemExit("Submitting uses one of today's leaderboard submissions. Re-run with --yes to confirm.")
    version: List[str] = ["-v", str(args.version)] if args.version else []
    kaggle("competitions", "submit", "-c", COMPETITION, "-k", kernel_ref(args.nb), "-f", "submission.csv",
           *version, "-m", args.message)


def cmd_scores(args) -> None:
    kaggle("competitions", "submissions", "-c", COMPETITION)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push-datasets"); p.add_argument("-m", "--message", default="RL77 rebuild"); p.add_argument("--with-candidates", action="store_true")
    p.set_defaults(fn=cmd_push_datasets)
    for name, fn in (("push", cmd_push), ("status", lambda a: print(status(a.nb))), ("fetch", cmd_fetch)):
        p = sub.add_parser(name); p.add_argument("nb", choices=list(NOTEBOOKS)); p.set_defaults(fn=fn)
    p = sub.add_parser("wait"); p.add_argument("nb", choices=list(NOTEBOOKS)); p.add_argument("--hours", type=float, default=12.5)
    p.add_argument("--poll", type=int, default=120); p.set_defaults(fn=cmd_wait)
    p = sub.add_parser("publish-output"); p.add_argument("nb", choices=list(NOTEBOOKS)); p.add_argument("-m", "--message", default="RL77 run outputs")
    p.set_defaults(fn=cmd_publish_output)
    p = sub.add_parser("submit"); p.add_argument("nb", choices=list(NOTEBOOKS)); p.add_argument("--version", type=int)
    p.add_argument("-m", "--message", default="RL77"); p.add_argument("--yes", action="store_true"); p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("scores"); p.set_defaults(fn=cmd_scores)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

"""Download the public files named by the blocked experiments.

Run from the repository root:

    /home/sweeden/.conda/envs/kagg/bin/python -m experiments.fetch_blockers

Files already at the target size are left in place. This does not train a
model and does not change the active blend.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]

DOWNLOADS = [
    {
        "path": "artifacts/dreams/ssl_model.ckpt",
        "url": "https://zenodo.org/api/records/10997887/files/ssl_model.ckpt/content",
        "bytes": 1392710212,
        "md5": "4deee91dd4486c2b6cbcde1f200b8f50",
        "note": "DreaMS self-supervised backbone (Zenodo 10997887). Not the contrastive embedding_model.ckpt.",
    },
    {
        "path": "artifacts/EN4/mist_msg_noaug.ckpt",
        "url": "https://zenodo.org/api/records/23303289/files/mist_msg_noaug.ckpt/content",
        "bytes": 289222619,
        "md5": "",
        "note": "MIST trained on MassSpecGym real spectra, no synthetic augmentation (Zenodo 23303289).",
    },
    {
        "path": "artifacts/DN6/checkpoints_duallgd.tar.gz",
        "url": "https://zenodo.org/api/records/20281201/files/checkpoints_duallgd.tar.gz/content",
        "bytes": 3216660853,
        "md5": "",
        "note": "Official DualLGD checkpoint archive (Zenodo 20281201). Not a TorchScript export.",
    },
    {
        "path": "artifacts/DN1/diffms_checkpoints.tar.gz",
        "url": "https://zenodo.org/api/records/15122968/files/diffms_checkpoints.tar.gz/content",
        "bytes": 939635008,
        "md5": "",
        "note": "DiffMS release used with its MIST encoder (Zenodo 15122968).",
    },
]


def _download(item: dict) -> None:
    target = REPO_ROOT / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item["bytes"])
    if target.is_file() and target.stat().st_size == expected:
        print(f"have {item['path']}", flush=True)
        return
    partial = target.with_suffix(target.suffix + ".partial")
    start = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(item["url"], headers={"User-Agent": "casmi-fetch/1"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    print(f"get {item['path']} from byte {start}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("ab") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            start += len(chunk)
            print(f"  {item['path']} {start}/{expected}", flush=True)
    if partial.stat().st_size != expected:
        raise SystemExit(f"{item['path']} is {partial.stat().st_size} bytes, expected {expected}")
    partial.replace(target)
    if item["md5"]:
        digest = hashlib.md5()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["md5"]:
            raise SystemExit(f"{item['path']} md5 {digest.hexdigest()} != {item['md5']}")
    sidecar = target.with_suffix(target.suffix + ".source.json")
    sidecar.write_text(json.dumps({k: item[k] for k in ("url", "bytes", "md5", "note")}, indent=2) + "\n")
    print(f"wrote {item['path']}", flush=True)


def _snapshots() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="ibm-research/GP-MoLFormer-Uniq",
        local_dir=str(REPO_ROOT / "artifacts" / "GM0" / "gp-molformer"),
    )
    print("wrote artifacts/GM0/gp-molformer", flush=True)
    snapshot_download(
        repo_id="DeepChem/ChemBERTa-77M-MLM",
        local_dir=str(REPO_ROOT / "artifacts" / "RR5" / "chemberta"),
    )
    print("wrote artifacts/RR5/chemberta", flush=True)
    note = REPO_ROOT / "artifacts" / "RR5" / "chemberta" / "TRAINING_DATA.txt"
    note.write_text(
        "DeepChem/ChemBERTa-77M-MLM is the 77M-parameter MLM trained on PubChem SMILES.\n"
        "The experiment allows a PubChem-pretrained SMILES encoder. ZINC-only ChemBERTa weights were not used.\n"
    )


def main() -> None:
    only = sys.argv[1:]
    items = [item for item in DOWNLOADS if not only or Path(item["path"]).name in only or item["path"] in only]
    for item in items:
        _download(item)
    if not only or "snapshots" in only:
        _snapshots()


if __name__ == "__main__":
    main()

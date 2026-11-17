# CASMI 2026 Workspace Agent Rules

## 1. Package & Environment Management (`uv` Required)
- **Always use `uv`** as the Python package and environment manager for all local installations, dependency resolution, and script execution.
- Do **not** run bare `pip`, `pip3`, or `conda install` locally.
- Target the workspace virtual environment (`/mnt/LMSTUDIO/CASMI/.venv`) using `uv`:
  - Install packages: `uv pip install --python /mnt/LMSTUDIO/CASMI/.venv/bin/python <packages>`
  - Run scripts/modules: `uv run --python /mnt/LMSTUDIO/CASMI/.venv/bin/python <command>` (or `/mnt/LMSTUDIO/CASMI/.venv/bin/python <command>`)

## 2. GPU Tasks & Training (`colab` Required)
- **Always run GPU tasks using Google Colab** (`colab` CLI at `~/.local/bin/colab` or the Colab notebooks in `notebooks/`). Never run heavy GPU training or neural model jobs on the local machine.
- Follow the workflow documented in `experiments/COLAB.md`:
  - Check quota first with `colab usage`.
  - Always name sessions with `-s <session-name>` (e.g., `colab new -s casmi-gpu --gpu T4` or `--gpu A100 --high-mem`) and pass `--timeout 0` to `colab exec` / `colab run` so long-running training steps do not time out after 30 seconds of silence.
  - Stage competition datasets and checkpoints on Google Drive (`/content/drive/MyDrive/CASMI2026`) and local Colab NVMe (`/content/casmi_fast`).
  - For one-shot GPU scripts, use `colab run --gpu <T4|A100> --timeout 0`.
  - Always stop the Colab session (`colab stop -s <session-name>`) after retrieving artifacts to end compute-unit charges.

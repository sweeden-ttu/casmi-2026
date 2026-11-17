# Colab CLI for GPU experiments

`colab` 0.8.0 is on `PATH` at `~/.local/bin/colab`. GPU types are `T4`, `L4`, `G4`, `A100`, and `H100`. A `400` from `colab new --gpu` means this account has no quota for that GPU; ask for `T4` instead. An unrecognized GPU name is treated as `A100`.

Run these from `/mnt/LMSTUDIO/CASMI`. Name every session with `-s`. `colab exec` and `colab run` fail after 30 seconds of silence unless you pass `--timeout 0`.

These runs stay on the competition data. `data/synthetic` is not a training set. Seed is `3407`. Leave the active blend in `experiments/weights/current.json` alone. NPC1 is the same checkpoint as NPC0, so it does not get its own GPU session. A finished run still calls the Evidently weight publish inside `python -m experiments`.

## See quota, then prove the GPU

```bash
colab usage
colab new -s casmi-gpu --gpu T4
colab status -s casmi-gpu
printf '%s\n' 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))' \
  | colab exec -s casmi-gpu --timeout 120
```

`T4` is the session to open first. Use `A100` for a full NPComposer or GP-MoLFormer train when `colab usage` shows that shape is available. Add `--high-mem` on CPU, T4, or A100 when the Pro plan allows it. L4 ignores that flag.

## Mount Drive and install what the VM lacks

The runtime path layout is `/content/drive/MyDrive/CASMI2026` and `/content/casmi_fast` (`casmi26/config.py`). Colab images already ship a CUDA build of PyTorch. Check `torch.cuda.is_available()` before installing `torch` again.

```bash
colab drivemount -s casmi-gpu
colab ssh -s casmi-gpu -- python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
colab install -s casmi-gpu pandas pyarrow scikit-learn transformers huggingface_hub
```

Put `train.parquet` and the DreaMS backbone on Drive, then copy them to `/content/casmi_fast`. Uploading the 2.8 GB parquet from this machine is the slow path.

## Train one blocked experiment

GPU queue: NPC0, GM0, DN6, DN1, P4, RR1, RR5, FH0. Each stays blocked until the artifact in `experiments/TRACKER.md` is on the VM. Open one named session per run and pass `--timeout 0`.

```bash
colab new -s casmi-npc0 --gpu A100 --high-mem
colab drivemount -s casmi-npc0
colab ssh -s casmi-npc0 -- bash -lc 'cd /content/casmi_fast && python -m experiments NPC0'
colab log -s casmi-npc0 -o experiments/runs/NPC0/colab.md
colab stop -s casmi-npc0
```

Same shape for the other trains. Change the session name and the experiment id:

```bash
colab ssh -s casmi-gm0 -- bash -lc 'cd /content/casmi_fast && python -m experiments GM0'
colab ssh -s casmi-dn6 -- bash -lc 'cd /content/casmi_fast && python -m experiments DN6'
colab ssh -s casmi-dn1 -- bash -lc 'cd /content/casmi_fast && python -m experiments DN1'
colab ssh -s casmi-p4  -- bash -lc 'cd /content/casmi_fast && python -m experiments P4'
colab ssh -s casmi-rr1 -- bash -lc 'cd /content/casmi_fast && python -m experiments RR1'
colab ssh -s casmi-rr5 -- bash -lc 'cd /content/casmi_fast && python -m experiments RR5'
colab ssh -s casmi-fh0 -- bash -lc 'cd /content/casmi_fast && python -m experiments FH0'
```

NPC0 initializes from `artifacts/NPC0/npcomposer`. GM0 initializes from `artifacts/GM0/gp-molformer`. Both need `artifacts/dreams/backbone.pt` first.

## One-shot GPU job

`colab run` creates the VM, runs the local file, and deletes the VM. `--timeout 0` keeps a quiet training step alive. `--keep` leaves the session up so a later `colab exec -s` can inspect it.

```bash
chmod +x scripts/colab_cuda_check.py
./scripts/colab_cuda_check.py
colab sessions
colab stop
```

`scripts/colab_cuda_check.py` already has the shebang `colab run --gpu T4 --timeout 0`. A training file uses the same shebang with the GPU that experiment needs:

```python
#!/usr/bin/env -S colab run --gpu A100 --high-mem --timeout 0 --keep
```

## Bring artifacts back, then release the GPU

```bash
colab ls -s casmi-npc0 /content/casmi_fast/experiments/runs/NPC0
colab download -s casmi-npc0 /content/casmi_fast/experiments/runs/NPC0/manifest.json experiments/runs/NPC0/manifest.json
colab ssh -s casmi-npc0 -- tar -C /content/casmi_fast -czf - experiments/runs/NPC0 \
  | tar -xzf - -C /mnt/LMSTUDIO/CASMI
colab stop -s casmi-npc0
colab sessions
```

`colab download` moves one file. Use the `tar` pipe for a run directory. `colab stop` is what ends the compute-unit charge.

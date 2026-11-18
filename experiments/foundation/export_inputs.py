"""Write the inputs every foundation-model run reads (local CPU, no GPU, no neural work).

    .venv/bin/python -m experiments.foundation.export_inputs [--train-spectra-per-mol 6] [--max-peaks 100]

Outputs in experiments/foundation/inputs/:
  train_spectra.parquet  nb1 training-split molecules (40,000), up to N spectra each, for heads trained on frozen
                         embeddings (RR1/IV1). No validation structure appears here (RL77's split is leakage-free).
  val_spectra.parquet    nb1's 998 held-out timsTOF molecules, RL77's filter (timsTOF, precursor 157-1159), <= 16 spectra.
  test_spectra.parquet   test.parquet (400 molecules, 1,213 spectra).
  candidates.parquet     every store structure in each val/test molecule's mass window (RL77 rule), with
                         source_tier and in_coconut.
  molecules.parquet      one row per val/test molecule: split, primary adduct, neutral mass, window size, and for val
                         the canonical truth key, truth_in_window and truth_in_coconut.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.foundation.common import (
    INPUTS_DIR, REPO_ROOT, RL77, list_column, neutral_mass, primary_index, window_bounds, write_json,
)

TRAIN_PARQUET = REPO_ROOT / "data" / "raw" / "train.parquet"
TEST_PARQUET = REPO_ROOT / "data" / "raw" / "test.parquet"
SPLIT_DIR = RL77 / "build" / "outputs" / "nb1" / "split"
CATALOG = RL77 / "build" / "outputs" / "nb2" / "candidate_catalog_sorted.parquet"
COCONUT = REPO_ROOT / "data" / "candidates" / "coconut" / "coconut_standardized.parquet"
COLUMNS = ["inchikey14", "normalized_smiles", "molecular_formula", "instrument_type", "ionization_mode", "adduct",
           "precursor_mz", "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]


def top_peaks(mzs, intens, k: int):
    mzs, intens = np.asarray(mzs, dtype=np.float32), np.asarray(intens, dtype=np.float32)
    if len(mzs) > k:
        keep = np.sort(np.argpartition(-intens, k)[:k])
        mzs, intens = mzs[keep], intens[keep]
    return mzs, intens


def mean_ce(value) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else np.nan
    except (TypeError, ValueError):
        return np.nan


def frame(df: pd.DataFrame, split: str, id_col: str, max_peaks: int) -> pd.DataFrame:
    peaks = [top_peaks(m, i, max_peaks) for m, i in zip(df["ms2_mzs"], df["ms2_normalized_intensities"])]
    return pd.DataFrame({
        "split": split,
        "molecule_id": df[id_col].astype(str).to_numpy(),
        "spectrum_idx": df.groupby(id_col, sort=False).cumcount().to_numpy(),
        "smiles": df["normalized_smiles"].to_numpy() if "normalized_smiles" in df else None,
        "formula": df["molecular_formula"].to_numpy() if "molecular_formula" in df else None,
        "adduct": df["adduct"].astype(str).to_numpy(),
        "ionization_mode": df["ionization_mode"].astype(str).to_numpy(),
        "precursor_mz": df["precursor_mz"].astype(np.float64).to_numpy(),
        "collision_energy": [mean_ce(v) for v in df["collision_energy_ev"]],
        "mzs": list_column([p[0] for p in peaks]),
        "intensities": list_column([p[1] for p in peaks]),
    })


def read_split(name: str):
    return [s.strip() for s in (SPLIT_DIR / f"{name}_inchikey14.txt").read_text().splitlines() if s.strip()]


def canonical_ik14(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize

    global _TE
    try:
        _TE
    except NameError:
        _TE = rdMolStandardize.TautomerEnumerator()
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    ik = inchi.MolToInchiKey(_TE.Canonicalize(mol))
    return ik[:14] if ik and len(ik) >= 14 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-spectra-per-mol", type=int, default=6)
    ap.add_argument("--val-spectra-per-mol", type=int, default=16)
    ap.add_argument("--max-peaks", type=int, default=100)
    args = ap.parse_args()
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    # Training molecules (all instruments: the frozen encoders are instrument-agnostic) and validation (RL77 filter)
    train_ids, val_ids = read_split("train"), read_split("val")
    train = pq.read_table(TRAIN_PARQUET, columns=COLUMNS, filters=[("inchikey14", "in", train_ids)]).to_pandas()
    train = train.dropna(subset=["normalized_smiles"]).groupby("inchikey14", sort=False).head(args.train_spectra_per_mol)
    train_out = frame(train.reset_index(drop=True), "train", "inchikey14", args.max_peaks)
    train_out.to_parquet(INPUTS_DIR / "train_spectra.parquet", index=False, compression="zstd")
    print(f"[export] train: {train_out['molecule_id'].nunique():,} molecules, {len(train_out):,} spectra ({time.time()-t0:.0f}s)")

    val = pq.read_table(TRAIN_PARQUET, columns=COLUMNS, filters=[("inchikey14", "in", val_ids)]).to_pandas()
    val = val[(val["instrument_type"] == "timsTOF") & val["precursor_mz"].between(157.0, 1159.0)]
    val = val.dropna(subset=["normalized_smiles"]).groupby("inchikey14", sort=False).head(args.val_spectra_per_mol)
    val_out = frame(val.reset_index(drop=True), "val", "inchikey14", args.max_peaks)
    val_out.to_parquet(INPUTS_DIR / "val_spectra.parquet", index=False, compression="zstd")

    test = pd.read_parquet(TEST_PARQUET)
    test_out = frame(test, "test", "molecule_id", args.max_peaks)
    test_out.to_parquet(INPUTS_DIR / "test_spectra.parquet", index=False, compression="zstd")
    print(f"[export] val: {val_out['molecule_id'].nunique():,} molecules / {len(val_out):,} spectra; "
          f"test: {test_out['molecule_id'].nunique():,} / {len(test_out):,} ({time.time()-t0:.0f}s)")

    # Candidate windows from the notebook-2 store (sorted by exact mass), RL77's window rule
    cat = pd.read_parquet(CATALOG, columns=["inchikey14", "canonical_smiles", "exact_mass", "molecular_formula", "source_tier"])
    coconut = set(pd.read_parquet(COCONUT, columns=["inchikey14"])["inchikey14"].dropna())
    masses = cat["exact_mass"].to_numpy()
    mol_rows, cand_parts = [], []
    for split, spec in (("val", val_out), ("test", test_out)):
        for mol_id, grp in spec.groupby("molecule_id", sort=False):
            p = primary_index(grp["adduct"].tolist())
            row = grp.iloc[p]
            mass = neutral_mass(row["precursor_mz"], row["adduct"], row["ionization_mode"])
            lo, hi = window_bounds(masses, mass)
            rec = {"split": split, "molecule_id": mol_id, "primary_adduct": row["adduct"], "neutral_mass": mass,
                   "n_candidates": hi - lo, "n_spectra": len(grp),
                   "n_positive_monomer": int(grp["adduct"].isin(["[M+H]+", "[M+Na]+", "[M+NH4]+", "[M+K]+"]).sum())}
            if split == "val":
                truth = canonical_ik14(grp["smiles"].iloc[0]) or mol_id
                window_iks = set(cat["inchikey14"].iloc[lo:hi])
                rec.update(truth_ik14=truth, truth_formula=grp["formula"].iloc[0], truth_in_window=truth in window_iks,
                           truth_in_coconut=truth in coconut)
            mol_rows.append(rec)
            part = cat.iloc[lo:hi].copy()
            part.insert(0, "molecule_id", mol_id)
            cand_parts.append(part)
    molecules = pd.DataFrame(mol_rows)
    cands = pd.concat(cand_parts, ignore_index=True).rename(columns={"canonical_smiles": "smiles", "molecular_formula": "formula"})
    cands["in_coconut"] = cands["inchikey14"].isin(coconut)
    molecules.to_parquet(INPUTS_DIR / "molecules.parquet", index=False)
    cands.to_parquet(INPUTS_DIR / "candidates.parquet", index=False, compression="zstd")

    v = molecules[molecules["split"] == "val"]
    stats = {
        "train_molecules": int(train_out["molecule_id"].nunique()), "train_spectra": int(len(train_out)),
        "val_molecules": int(len(v)), "test_molecules": int((molecules["split"] == "test").sum()),
        "candidate_rows": int(len(cands)), "unique_candidates": int(cands["inchikey14"].nunique()),
        "candidates_per_molecule_median": float(molecules["n_candidates"].median()),
        "candidates_per_molecule_p95": float(molecules["n_candidates"].quantile(0.95)),
        "val_truth_in_window": float(v["truth_in_window"].mean()),
        "val_truth_in_coconut": float(v["truth_in_coconut"].mean()),
        "test_molecules_with_positive_monomer": float((molecules.loc[molecules["split"] == "test", "n_positive_monomer"] > 0).mean()),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    write_json(INPUTS_DIR / "export_stats.json", stats)
    print(f"[export] {stats}")


if __name__ == "__main__":
    main()

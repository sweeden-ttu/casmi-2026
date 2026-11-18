# [Colab Cell 4] Multi-Spectrum Molecule Dataset & Leakage-Free InChIKey14 Split
def create_leakage_free_splits(
    df: pd.DataFrame, val_fraction: float = 0.10, seed: int = CFG.SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits training data so zero inchikey14 connectivity blocks overlap between train and val,
    while prioritizing timsTOF spectra (157-1159 Da) for validation alignment with test.parquet.
    """
    rng = np.random.default_rng(seed)
    timstof_mask = (
        (df["instrument_type"] == "timsTOF")
        & (df["precursor_mz"] >= 157.0)
        & (df["precursor_mz"] <= 1159.0)
    )
    eligible_ik14 = df.loc[timstof_mask, "inchikey14"].dropna().unique()
    if len(eligible_ik14) == 0:
        eligible_ik14 = df["inchikey14"].dropna().unique()

    n_val = max(1, int(len(eligible_ik14) * val_fraction))
    val_ik14_set = set(rng.choice(eligible_ik14, size=n_val, replace=False))

    val_df = df[df["inchikey14"].isin(val_ik14_set) & timstof_mask].reset_index(drop=True)
    train_df = df[~df["inchikey14"].isin(val_ik14_set)].reset_index(drop=True)
    return train_df, val_df


TRAIN_COLUMNS = [
    "inchikey14", "normalized_smiles", "molecular_formula", "ingest_lib", "instrument_type",
    "ionization_mode", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
    "base_peak_intensity", "collision_energy_ev",
]


def load_competition_frames(
    data_dir: Path,
    n_train_mols: int,
    n_val_mols: int,
    seed: int = CFG.SEED,
    max_spectra_per_mol: int = CFG.MAX_SPECTRA_LOADED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Streams train.parquet (2.5M spectra, 3 GB) with the same leakage rule as
    create_leakage_free_splits, without materializing the whole file in pandas:
      - validation inchikey14 blocks are sampled from timsTOF spectra at 157-1159 Da,
      - those blocks are removed from training entirely,
      - training blocks are sampled from the rest.
    Returns (train_df, val_df, library_df) where library_df holds one SMILES per inchikey14
    over the whole file (the retrieval library used for validation and test).
    """
    import pyarrow.compute as pc

    path = Path(data_dir) / "train.parquet"
    keys = pq.read_table(path, columns=["inchikey14", "normalized_smiles", "instrument_type", "precursor_mz"])
    timstof = pc.and_(
        pc.equal(keys["instrument_type"], "timsTOF"),
        pc.and_(pc.greater_equal(keys["precursor_mz"], 157.0), pc.less_equal(keys["precursor_mz"], 1159.0)),
    )
    eligible = np.array(sorted(pc.unique(pc.filter(keys["inchikey14"], timstof)).to_pylist()))
    all_ik14 = np.array(sorted(pc.unique(keys["inchikey14"]).to_pylist()))

    rng = np.random.default_rng(seed)
    val_ik14 = set(rng.choice(eligible, size=min(n_val_mols, len(eligible)), replace=False).tolist())
    remaining = np.array([k for k in all_ik14 if k not in val_ik14])
    train_ik14 = set(rng.choice(remaining, size=min(n_train_mols, len(remaining)), replace=False).tolist())

    lib = (
        keys.select(["inchikey14", "normalized_smiles"]).to_pandas()
        .rename(columns={"normalized_smiles": "smiles"})
        .dropna().drop_duplicates("inchikey14").reset_index(drop=True)
    )
    del keys

    selected = sorted(val_ik14 | train_ik14)
    table = pq.read_table(path, columns=TRAIN_COLUMNS, filters=[("inchikey14", "in", selected)])
    df = table.to_pandas()
    del table
    df = df.groupby("inchikey14", sort=False).head(max_spectra_per_mol).reset_index(drop=True)

    in_window = (
        (df["instrument_type"] == "timsTOF") & (df["precursor_mz"] >= 157.0) & (df["precursor_mz"] <= 1159.0)
    )
    val_df = df[df["inchikey14"].isin(val_ik14) & in_window].reset_index(drop=True)
    train_df = df[df["inchikey14"].isin(train_ik14)].reset_index(drop=True)
    assert not set(train_df["inchikey14"]) & set(val_df["inchikey14"]), "inchikey14 leakage between train and val"
    ENV.log(
        f"Loaded {train_df['inchikey14'].nunique():,} train molecules ({len(train_df):,} spectra), "
        f"{val_df['inchikey14'].nunique():,} val molecules ({len(val_df):,} spectra), library {len(lib):,} structures"
    )
    return train_df, val_df, lib


def load_spectra_for_inchikeys(
    data_dir: Path,
    inchikeys: List[str],
    timstof_window_only: bool = False,
    max_spectra_per_mol: int = CFG.MAX_SPECTRA_LOADED,
) -> pd.DataFrame:
    """Reloads the spectra of a saved split (notebook 1 `split/*.txt`) so notebooks 2-5 use the same molecules."""
    table = pq.read_table(Path(data_dir) / "train.parquet", columns=TRAIN_COLUMNS,
                          filters=[("inchikey14", "in", sorted(set(inchikeys)))])
    df = table.to_pandas()
    if timstof_window_only:
        df = df[(df["instrument_type"] == "timsTOF") & (df["precursor_mz"] >= 157.0) & (df["precursor_mz"] <= 1159.0)]
    df = df.dropna(subset=["normalized_smiles"])
    return df.groupby("inchikey14", sort=False).head(max_spectra_per_mol).reset_index(drop=True)


def read_split(split_dir: Path, name: str) -> List[str]:
    return [line.strip() for line in (Path(split_dir) / f"{name}_inchikey14.txt").read_text().splitlines() if line.strip()]


def ground_truth_ik14(df: pd.DataFrame) -> Dict[str, str]:
    """Validation answer key: molecule_id -> InChIKey14 of the tautomer-canonical true structure.

    train.parquet has no molecule_id, so validation groups (and records) are keyed by its stored inchikey14. The scorer
    compares tautomer-canonical InChIKey14s, and 0.3% of the held-out molecules (1.5% of train) store the key of a
    non-canonical tautomer, which `{mid: mid}` made unmatchable even for a correct answer."""
    first = df.drop_duplicates("inchikey14")
    return {str(mid): (canonicalize_and_inchikey14(smi)[1] or str(mid))
            for mid, smi in zip(first["inchikey14"], first["normalized_smiles"])}


def _finite_or(value, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _primary_record_index(records: List[dict]) -> int:
    """First spectrum whose adduct is a known singly charged monomer, so precursor_mz gives M directly."""
    for i, row in enumerate(records):
        spec = ADDUCT_SPECS.get(str(row.get("adduct", "")))
        if spec is not None and spec[0] == 1 and abs(spec[1]) == 1:
            return i
    return 0


class MultiSpectrumMoleculeDataset(Dataset):
    """
    Groups all MS/MS spectra belonging to the same molecule_id (1 to 16 spectra per molecule)
    for joint multi-spectrum encoding and contrastive/generative training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: SMILESTokenizer,
        max_spectra_per_mol: int = 8,
        is_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_spectra_per_mol = max_spectra_per_mol
        self.is_train = is_train

        group_col = "molecule_id" if "molecule_id" in df.columns else "inchikey14"
        self.groups = []
        for mol_id, grp in df.groupby(group_col, sort=False):
            self.groups.append((str(mol_id), grp.to_dict("records")))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mol_id, records = self.groups[idx]
        if self.is_train and len(records) > self.max_spectra_per_mol:
            records = random.sample(records, self.max_spectra_per_mol)
        else:
            records = records[: self.max_spectra_per_mol]

        mzs_list, ints_list, nls_list, ce_list, adduct_list, bpi_list = [], [], [], [], [], []
        for row in records:
            cached = row.get("_preprocessed")
            if cached is None:
                is_enveda = str(row.get("ingest_lib", "")) == "enveda-180"
                cached = preprocess_spectrum(
                    row["ms2_mzs"],
                    row["ms2_normalized_intensities"],
                    float(row["precursor_mz"]),
                    ionization_mode=str(row.get("ionization_mode", "positive")),
                    apply_electron_mass_correction=(self.is_train and is_enveda),
                )
                row["_preprocessed"] = cached  # per-worker cache; DataLoader workers are persistent
            mzs, ints, nls = cached
            mzs_list.append(mzs)
            ints_list.append(ints)
            nls_list.append(nls)
            ce_list.append(harmonize_collision_energy(row.get("collision_energy_ev", 35.0)))
            adduct_list.append(ADDUCT_VOCAB.get(str(row.get("adduct", "UNKNOWN")), ADDUCT_VOCAB["UNKNOWN"]))
            # 46% of train.parquet base_peak_intensity values are null; max(nan, 1.0) is nan, so guard first.
            bpi = _finite_or(row.get("base_peak_intensity"), 1e4)
            bpi_list.append(math.log10(max(bpi, 1.0)))

        # Pad spectrum count up to max_spectra_per_mol
        n_spec = len(mzs_list)
        spec_mask = np.zeros(self.max_spectra_per_mol, dtype=bool)
        spec_mask[:n_spec] = True

        while len(mzs_list) < self.max_spectra_per_mol:
            mzs_list.append(np.zeros(CFG.MAX_PEAKS, dtype=np.float32))
            ints_list.append(np.zeros(CFG.MAX_PEAKS, dtype=np.float32))
            nls_list.append(np.zeros(CFG.MAX_PEAKS, dtype=np.float32))
            ce_list.append(0.0)
            adduct_list.append(ADDUCT_VOCAB["UNKNOWN"])
            bpi_list.append(0.0)

        item = {
            "molecule_id": mol_id,
            "mzs": torch.tensor(np.stack(mzs_list), dtype=torch.float32),          # (S, P)
            "ints": torch.tensor(np.stack(ints_list), dtype=torch.float32),        # (S, P)
            "nls": torch.tensor(np.stack(nls_list), dtype=torch.float32),          # (S, P)
            "ce": torch.tensor(ce_list, dtype=torch.float32),                      # (S,)
            "adduct": torch.tensor(adduct_list, dtype=torch.long),                 # (S,)
            "bpi": torch.tensor(bpi_list, dtype=torch.float32),                    # (S,)
            "spec_mask": torch.tensor(spec_mask, dtype=torch.bool),                # (S,)
        }
        primary = records[_primary_record_index(records)]
        precursor_mz = float(primary["precursor_mz"])
        if self.is_train and str(primary.get("ingest_lib", "")) == "enveda-180" \
                and str(primary.get("ionization_mode", "positive")).lower() == "positive":
            precursor_mz -= CFG.ELECTRON_MASS_CORR  # same correction preprocess_spectrum applied
        item["precursor_mz"] = torch.tensor(precursor_mz, dtype=torch.float64)
        item["primary_adduct"] = str(primary.get("adduct", "[M+H]+"))
        item["ionization_mode"] = str(primary.get("ionization_mode", "positive"))

        smiles = records[0].get("normalized_smiles")
        if isinstance(smiles, str) and smiles:
            item["mol_tokens_cls"] = self.tokenizer.encode(smiles, add_cls=True)
            item["mol_tokens_dec"] = self.tokenizer.encode(smiles, add_cls=False)
            item["smiles"] = smiles
            item["inchikey14"] = str(records[0].get("inchikey14", ""))
            item["molecular_formula"] = str(records[0].get("molecular_formula", "") or "")

        return item
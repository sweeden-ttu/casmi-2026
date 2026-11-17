# [Colab Cell 2] Chemical standardization, evaluation metric (MRR@25), and SMILES vocabulary
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

PROTON = 1.007276467
ELECTRON = 0.000548580

# Adduct name -> (molecules per ion n, charge z, mass added to n*M to form the ion).
# Names follow train.parquet / test.parquet spelling (formate is "[M+CH2O2-H]-", acetate "[M+C2H4O2-H]-").
ADDUCT_SPECS: Dict[str, Tuple[int, int, float]] = {
    "[M+H]+": (1, 1, PROTON),
    "[M+Na]+": (1, 1, 22.989218),
    "[M+NH4]+": (1, 1, 18.033823),
    "[M+K]+": (1, 1, 38.963158),
    "[M-H2O+H]+": (1, 1, PROTON - 18.010565),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * 18.010565),
    "[M]+": (1, 1, -ELECTRON),
    "[M-H]-": (1, -1, -PROTON),
    "[M+CH2O2-H]-": (1, -1, 46.005479 - PROTON),
    "[M+C2H4O2-H]-": (1, -1, 60.021129 - PROTON),
    "[M+Cl]-": (1, -1, 34.969402),
    "[2M+H]+": (2, 1, PROTON),
    "[2M+Na]+": (2, 1, 22.989218),
    "[2M-H]-": (2, -1, -PROTON),
    "[M+2H]2+": (1, 2, 2 * PROTON),
    "[M-2H]2-": (1, -2, -2 * PROTON),
}
ADDUCT_VOCAB = {name: idx for idx, name in enumerate(ADDUCT_SPECS)}
ADDUCT_VOCAB["UNKNOWN"] = len(ADDUCT_VOCAB)

# Shifts for singly charged monomer adducts, kept for callers that use neutral = precursor_mz + shift.
ADDUCT_MASS_SHIFT = {name: -delta for name, (n, z, delta) in ADDUCT_SPECS.items() if n == 1 and abs(z) == 1}


def neutral_mass_from_precursor(precursor_mz: float, adduct: str, ionization_mode: str = "positive") -> float:
    """Neutral monoisotopic mass M from precursor m/z: M = (|z| * m/z - delta) / n."""
    spec = ADDUCT_SPECS.get(str(adduct))
    if spec is None:
        spec = ADDUCT_SPECS["[M-H]-"] if str(ionization_mode).lower().startswith("neg") else ADDUCT_SPECS["[M+H]+"]
    n, z, delta = spec
    return (abs(z) * float(precursor_mz) - delta) / n


@lru_cache(maxsize=2_000_000)
def canonicalize_and_inchikey14(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Passes SMILES through RDKit tautomer canonicalization and extracts
    the 14-character planar connectivity block (InChIKey14). Cached: the same
    candidate SMILES is scored many times across molecules and ranking passes.
    """
    if not smiles or not isinstance(smiles, str):
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        canon_mol = tautomer_enumerator.Canonicalize(mol)
        canon_smiles = Chem.MolToSmiles(canon_mol, isomericSmiles=False, canonical=True)
        ik = inchi.MolToInchiKey(canon_mol)
        if not ik or len(ik) < 14:
            return None, None
        return canon_smiles, ik[:14]
    except Exception:
        return None, None


def compute_casmi_mrr25(
    predictions: Dict[str, List[str]],
    ground_truth_ik14: Dict[str, str]
) -> float:
    """Computes official first-match MRR@25 across molecules using InChIKey14."""
    reciprocal_ranks = []
    for mol_id, true_ik14 in ground_truth_ik14.items():
        preds = predictions.get(mol_id, [])[:CFG.TOP_N_SUBMISSION]
        rr = 0.0
        for rank_idx, smi in enumerate(preds, start=1):
            _, pred_ik14 = canonicalize_and_inchikey14(smi)
            if pred_ik14 is not None and pred_ik14 == true_ik14:
                rr = 1.0 / rank_idx
                break
        reciprocal_ranks.append(rr)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


class SMILESTokenizer:
    """Character tokenizer with [PAD], [CLS], [BOS], [EOS], and [UNK] tokens.

    CHARS is the character set of train.parquet normalized_smiles (no stereo), plus the
    stereo/bond characters so isomeric inputs do not collapse to [UNK]. Two-letter
    elements (Cl, Br, Si, As, Se, Na) are spelled as two character tokens.
    """
    SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[BOS]", "[EOS]", "[UNK]"]
    CHARS = list("#%()+-0123456789=ABCFHIKNOPS[]aceilnoprsg@/\\.")

    def __init__(self, max_len: int = CFG.MAX_SMILES_LEN):
        assert len(set(self.CHARS)) == len(self.CHARS), "duplicate characters in SMILES vocabulary"
        self.max_len = max_len
        self.vocab = self.SPECIAL_TOKENS + self.CHARS
        self.stoi = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.itos = {idx: ch for idx, ch in enumerate(self.vocab)}
        self.pad_id = self.stoi["[PAD]"]
        self.cls_id = self.stoi["[CLS]"]
        self.bos_id = self.stoi["[BOS]"]
        self.eos_id = self.stoi["[EOS]"]
        self.unk_id = self.stoi["[UNK]"]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, smiles: str, add_cls: bool = True) -> torch.Tensor:
        tokens = [self.cls_id] if add_cls else [self.bos_id]
        for ch in smiles[: self.max_len - 2]:
            tokens.append(self.stoi.get(ch, self.unk_id))
        tokens.append(self.eos_id)
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens.extend([self.pad_id] * pad_len)
        return torch.tensor(tokens[: self.max_len], dtype=torch.long)

    def decode(self, token_ids: List[int]) -> str:
        chars = []
        for idx in token_ids:
            idx = int(idx)
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.cls_id, self.bos_id, self.unk_id):
                continue
            chars.append(self.itos.get(idx, ""))
        return "".join(chars)

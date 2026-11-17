# [Kaggle Submission Cell] Zero-Copy Memory-Mapped Weight + Formula Pre-Retriever
class OfflineMassAndFormulaRetriever:
    """
    Zero-copy offline retriever for Kaggle submissions:
      1. Queries exact molecular formula bucket if a high-confidence formula is provided (MIST-CF / SIRIUS).
      2. Queries contiguous monoisotopic mass window [mass - tol, mass + tol] via O(log N) binary search.
      3. Falls back to FAISS IVF search if fewer than K=40 candidates match the mass window.
    """

    def __init__(
        self,
        dataset_dir: Path = Path("/kaggle/input/casmi26-glmr-offline-store"),
        d_model: int = 256,
    ):
        self.catalog_df = pd.read_parquet(dataset_dir / "candidate_catalog_sorted.parquet")
        self.smiles_arr = self.catalog_df["canonical_smiles"].to_numpy()
        self.ik14_arr = self.catalog_df["inchikey14"].to_numpy()
        self.masses_arr = self.catalog_df["exact_mass"].to_numpy(dtype=np.float64)
        self.source_tier_arr = self.catalog_df["source_tier"].to_numpy(dtype=np.uint8)

        n_rows = len(self.catalog_df)
        self.mmap_embs = np.memmap(
            dataset_dir / "chemformer_embs_fp16.mmap",
            dtype=np.float16,
            mode="r",
            shape=(n_rows, d_model),
        )

        self.faiss_index = faiss.read_index(str(dataset_dir / "chemformer_ivfflat.index"))
        if hasattr(self.faiss_index, "nprobe"):
            self.faiss_index.nprobe = 64

        # Load CSR formula index
        f_data = np.load(dataset_dir / "formula_offsets.npz", allow_pickle=False)
        formulas = f_data["formulas"]
        offsets = f_data["offsets"]
        flat_indices = f_data["flat_indices"]
        self.formula_to_span: Dict[str, Tuple[int, int]] = {
            str(f): (int(offsets[i]), int(offsets[i + 1])) for i, f in enumerate(formulas)
        }
        self.formula_flat_indices = flat_indices

    def get_candidate_indices(
        self,
        est_neutral_mass: float,
        ppm_tol: float = 15.0,
        predicted_formulas: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Returns catalog row indices constrained by formula priors and/or ppm mass window."""
        matched_indices = []

        # Priority A: Formula-based library candidates (GLMR Table 1 & 2)
        if predicted_formulas:
            for form in predicted_formulas:
                span = self.formula_to_span.get(form)
                if span is not None:
                    matched_indices.append(self.formula_flat_indices[span[0] : span[1]])

        # Priority B: Weight-based library candidates via O(log N) binary search
        tol_da = max(est_neutral_mass * (ppm_tol * 1e-6), 0.015)
        left_idx = int(np.searchsorted(self.masses_arr, est_neutral_mass - tol_da, side="left"))
        right_idx = int(np.searchsorted(self.masses_arr, est_neutral_mass + tol_da, side="right"))

        if right_idx > left_idx:
            matched_indices.append(np.arange(left_idx, right_idx, dtype=np.int32))

        if not matched_indices:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(matched_indices))

    @torch.no_grad()
    def retrieve_top_k_batch(
        self,
        e_s_batch: torch.Tensor,
        precursor_mzs: torch.Tensor,
        adducts: List[str],
        adduct_mass_shift: Dict[str, float],
        top_k: int = 40,
        ppm_tol: float = 15.0,
        in_library_boost: float = 0.03,
        formula_priors_batch: Optional[List[List[str]]] = None,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        Retrieves top-K candidate embeddings H^m_K per molecule directly from the FP16 memmap.
        Applies a small prior boost to Tier-1 in-library compounds when spectral similarity ties.
        """
        device = e_s_batch.device
        e_s_np = e_s_batch.detach().cpu().numpy().astype(np.float32)

        batch_embs = []
        batch_cat_ids = []

        for b in range(len(e_s_np)):
            prec_mz = float(precursor_mzs[b].item())
            shift = adduct_mass_shift.get(adducts[b], -1.007276)
            neutral_mass = prec_mz + shift
            f_priors = formula_priors_batch[b] if formula_priors_batch else None

            cand_ids = self.get_candidate_indices(neutral_mass, ppm_tol, f_priors)

            if len(cand_ids) >= top_k:
                # Slice directly from disk memmap without loading full array into RAM
                cand_vecs = np.asarray(self.mmap_embs[cand_ids], dtype=np.float32)
                scores = cand_vecs @ e_s_np[b]

                # Add slight tie-breaking prior for known training library compounds (source_tier == 1)
                tier1_mask = (self.source_tier_arr[cand_ids] == 1).astype(np.float32)
                scores = scores + in_library_boost * tier1_mask

                top_local = np.argpartition(-scores, kth=top_k - 1)[:top_k]
                top_local = top_local[np.argsort(-scores[top_local])]
                chosen_ids = cand_ids[top_local]
                chosen_vecs = cand_vecs[top_local]
            else:
                # Fallback to global FAISS inner-product search if mass window has < K isomers
                _, faiss_ids = self.faiss_index.search(e_s_np[b : b + 1], top_k)
                chosen_ids = faiss_ids[0].astype(np.int32)
                chosen_vecs = np.asarray(self.mmap_embs[chosen_ids], dtype=np.float32)

            batch_cat_ids.append(chosen_ids.tolist())
            batch_embs.append(torch.from_numpy(chosen_vecs).to(device))

        return torch.stack(batch_embs, dim=0), batch_cat_ids
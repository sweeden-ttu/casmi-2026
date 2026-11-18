# [Colab Ensemble Cell 4] Multi-Paradigm Blender: GLMR Stage 2 + Siamese ECFP4 + Mass2SMILES 60-FG
from scipy.optimize import minimize


class CASMI26EnsembleReranker:
    """
    Combines four orthogonal similarity signals for each candidate i in {1..K}:
      1. s_glmr_gen:  cos(E^m_+, E^m_i)           [GLMR Stage 2 Generative Unimodal Similarity]
      2. s_glmr_pre:  cos(E^s_mol, E^m_i)         [GLMR Stage 1 Contrastive Spectral Similarity]
      3. s_siamese:   0.5*cos(z_s, z_f,i) + 0.5*Tanimoto(p_hat, ecfp4_i) [2000-Bin Siamese ECFP4]
      4. s_fg_match:  exp(-gamma * ||fg_hat - fg_i||_1)                  [Mass2SMILES 71-D FG Kernel]
    """

    def __init__(
        self,
        weights: Optional[Tuple[float, float, float, float, float]] = None,
        fg_gamma: float = W("nb3.reranker.fg_gamma"),
        rrf_k: float = W("nb3.reranker.rrf_k"),
    ):
        if weights is None:  # Evidently snapshot (casmi-rl77-weights) instead of the PDF literals
            weights = tuple(W.vec("nb3.reranker", ["w_gen", "w_pre", "w_siamese", "w_fg", "w_rrf"]))
        # Weights: [w_glmr_gen, w_glmr_pre, w_siamese_ecfp4, w_mass2smiles_fg, w_rrf]
        self.weights = np.asarray(weights, dtype=np.float64)
        self.fg_gamma = fg_gamma
        self.rrf_k = rrf_k

    @staticmethod
    def _zscore_normalize(scores: np.ndarray) -> np.ndarray:
        if len(scores) <= 1:
            return scores
        std = float(np.std(scores))
        if std < 1e-7:
            return np.zeros_like(scores)
        return (scores - float(np.mean(scores))) / std

    def _reciprocal_rank_fusion(self, score_matrices: List[np.ndarray]) -> np.ndarray:
        """Computes consensus Reciprocal Rank Fusion (RRF) across the 4 scoring paradigms."""
        k_cands = len(score_matrices[0])
        rrf_scores = np.zeros(k_cands, dtype=np.float64)
        for s_vec in score_matrices:
            ranks = np.argsort(np.argsort(-s_vec)) + 1  # 1-indexed rank
            rrf_scores += 1.0 / (self.rrf_k + ranks)
        return self._zscore_normalize(rrf_scores)

    def score_candidates_for_molecule(
        self,
        s_glmr_gen: np.ndarray,      # (K,)
        s_glmr_pre: np.ndarray,      # (K,)
        s_siamese_ecfp4: np.ndarray, # (K,)
        s_fg_l1_dist: np.ndarray,    # (K,) mean L1 distance across 71 FG/heteroatom targets
        gen_smiles_valid: bool = True,
    ) -> np.ndarray:
        """Returns blended ranking score for all K candidates of a single molecule_id."""
        s_fg_sim = np.exp(-self.fg_gamma * s_fg_l1_dist)

        z_gen = self._zscore_normalize(s_glmr_gen) if gen_smiles_valid else np.zeros_like(s_glmr_pre)
        z_pre = self._zscore_normalize(s_glmr_pre)
        z_sia = self._zscore_normalize(s_siamese_ecfp4)
        z_fg = self._zscore_normalize(s_fg_sim)

        active_streams = [z_pre, z_sia, z_fg]
        if gen_smiles_valid:
            active_streams.append(z_gen)
        z_rrf = self._reciprocal_rank_fusion(active_streams)

        w0, w1, w2, w3, w4 = self.weights
        if not gen_smiles_valid:
            # Redistribute generative weight to Stage 1 and Siamese ECFP4 if beam SMILES failed RDKit check
            w1 += W("nb3.reranker.gen_fail_to_pre") * w0
            w2 += W("nb3.reranker.gen_fail_to_siamese") * w0
            w0 = 0.0

        return w0 * z_gen + w1 * z_pre + w2 * z_sia + w3 * z_fg + w4 * z_rrf

    def optimize_weights_oof(
        self,
        oof_records: List[Dict],
        ground_truth_ik14: Dict[str, str],
    ) -> np.ndarray:
        """
        Optimizes ensemble weights [w_gen, w_pre, w_sia, w_fg, w_rrf] on out-of-fold
        timsTOF validation molecules using Nelder-Mead simplex search to maximize MRR@25.
        """

        def neg_mrr25_objective(raw_w: np.ndarray) -> float:
            w = np.clip(raw_w, 0.0, 1.0)
            if np.sum(w) <= 1e-6:
                return 0.0
            self.weights = w / np.sum(w)

            rr_list = []
            for rec in oof_records:
                mol_id = rec["molecule_id"]
                true_ik14 = ground_truth_ik14.get(mol_id)
                if not true_ik14:
                    continue
                blended = self.score_candidates_for_molecule(
                    rec["s_glmr_gen"],
                    rec["s_glmr_pre"],
                    rec["s_siamese_ecfp4"],
                    rec["s_fg_l1_dist"],
                    gen_smiles_valid=rec["gen_valid"],
                )
                in_window = np.asarray(rec.get("in_window", np.zeros(len(blended), dtype=bool)), dtype=bool)
                order = np.lexsort((-blended, ~in_window))  # mass-consistent candidates first

                seen_ik14 = set()
                rank = 0
                matched_rr = 0.0
                for idx in order:
                    ik14 = rec["cand_ik14s"][idx]
                    if not ik14 or ik14 in seen_ik14:
                        continue
                    seen_ik14.add(ik14)
                    rank += 1
                    if ik14 == true_ik14:
                        matched_rr = 1.0 / rank
                        break
                    if rank >= 25:
                        break
                rr_list.append(matched_rr)

            return -float(np.mean(rr_list))

        # MRR@25 is piecewise constant in the weights, so one Nelder-Mead run from the prior can stall below
        # a trivial corner (Kaggle nb3 v1: one fold stopped at 0.222 while the Siamese stream alone scored ~0.37).
        # Score the prior, uniform and every single-stream vector, then search from the two best starts with an
        # explicit 0.2 simplex step (SciPy's default step for a zero entry is 0.00025, so one-hot starts never move).
        n = len(self.weights)
        starts = [np.asarray(self.weights, dtype=np.float64), np.full(n, 1.0 / n)] + [np.eye(n)[i] for i in range(n)]
        scored = sorted((neg_mrr25_objective(s), i) for i, s in enumerate(starts))  # ties keep the prior (index 0)
        best_fun, best_w = scored[0][0], starts[scored[0][1]]
        for _, i in scored[:2]:
            x0 = starts[i]
            simplex = np.vstack([x0] + [x0 + 0.2 * np.eye(n)[j] for j in range(n)])
            res = minimize(
                neg_mrr25_objective,
                x0,
                method="Nelder-Mead",
                options={"maxiter": 250, "xatol": 1e-3, "fatol": 1e-4, "initial_simplex": simplex},
            )
            if res.fun < best_fun:
                best_fun, best_w = res.fun, res.x
        best_w = np.clip(best_w, 0.0, 1.0)
        self.weights = best_w / np.sum(best_w)
        print(f"Optimized OOF MRR@25: {-best_fun:.4f} | Weights: {np.round(self.weights, 4)}")
        return self.weights
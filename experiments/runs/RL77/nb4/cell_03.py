# [Colab Formula Filter Cell 3] Fast Multi-Spectrum Precursor + Neutral Loss + Isotope + Neural Heteroatom Scorer
class FastMISTSubformulaFilter:
    """
    Ranks candidate molecular formulas for each molecule_id using 5 fast vectorized signals:
      1. Multi-Spectrum Consensus Mass PPM Likelihood (sigma_ppm = 4.5 ppm on timsTOF)
      2. MS/MS Diagnostic Neutral Loss Subformula Compatibility (SIRIUS/MIST-CF loss tree)
      3. Raw MS/MS Fine-Isotope & Halogen Pattern Match (37Cl, 81Br, 34S spacing at ~1.9975 Da)
      4. Mass2SMILES 71-D Neural Heteroatom Count Consistency (N, O, S, P, F, Cl, Br, I)
      5. RDBE vs. Predicted Aromatic + Aliphatic Ring Lower Bound (fg_71_pred[:, 69:71])
    """

    SCORE_KEYS = ("mass", "nl", "iso", "neural")

    def __init__(
        self,
        formula_bank: CandidateFormulaBank,
        adduct_mass_shift: Optional[Dict[str, float]] = None,
        ppm_sigma: float = W("nb4.filter.ppm_sigma"),
        nl_tol_da: float = W("nb4.filter.nl_tol_da"),
        score_weights: Optional[Dict[str, float]] = None,
    ):
        self.bank = formula_bank
        self.adduct_mass_shift = adduct_mass_shift  # kept for the PDF signature; see infer_consensus_neutral_mass
        self.ppm_sigma = ppm_sigma
        self.nl_tol_da = nl_tol_da
        # Composite posterior weights (PDF: 1.50 / 1.00 / 0.80 / 1.20) from the Evidently snapshot
        self.score_weights = dict(score_weights or {k: W(f"nb4.score.{k}") for k in self.SCORE_KEYS})

    def infer_consensus_neutral_mass(
        self,
        precursor_mzs: List[float],
        adducts: List[str],
        bpis: Optional[List[float]] = None,
        ionization_modes: Optional[List[str]] = None,
    ) -> float:
        """
        Combines precursor_mz measurements across all spectra of a molecule_id
        (accounting for adduct mass shifts) weighted by log10(base_peak_intensity).
        Uses neutral_mass_from_precursor(): the shift table missed formate/acetate spellings and [2M+X].
        """
        masses, weights = [], []
        for idx, (pmz, add) in enumerate(zip(precursor_mzs, adducts)):
            if pmz <= 0.0:
                continue
            mode = ionization_modes[idx] if ionization_modes else "positive"
            masses.append(neutral_mass_from_precursor(float(pmz), str(add), mode))
            w = math.log10(max(float(bpis[idx]), 10.0)) if bpis else 1.0
            weights.append(w)

        if not masses:
            return 0.0
        masses_arr = np.asarray(masses, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)

        # Reject outlier adduct misassignments (> 0.05 Da from median)
        med = float(np.median(masses_arr))
        inliers = np.abs(masses_arr - med) <= 0.05
        if np.any(inliers):
            return float(np.average(masses_arr[inliers], weights=weights_arr[inliers]))
        return med

    @staticmethod
    def detect_halogen_and_sulfur_isotopes(
        raw_mzs: np.ndarray, raw_ints: np.ndarray
    ) -> Tuple[bool, bool]:
        """
        Inspects raw peak pairs separated by 1.9955 - 1.9985 Da (M+2 isotope spacing for
        37Cl, 81Br, and 34S) among intense peaks to flag likely Cl/Br or S presence.
        """
        if len(raw_mzs) < 2:
            return False, False
        top_mask = raw_ints >= 0.10
        m = raw_mzs[top_mask]
        intens = raw_ints[top_mask]
        if len(m) < 2:
            return False, False

        diffs = np.abs(m[:, None] - m[None, :])
        m2_pairs = np.where((diffs >= 1.9950) & (diffs <= 1.9988))
        has_cl_br = False
        has_s = False
        for i, j in zip(*m2_pairs):
            if m[j] > m[i] and intens[i] > 0.15:
                ratio = intens[j] / intens[i]
                if 0.22 <= ratio <= 1.35:
                    has_cl_br = True
                elif 0.035 <= ratio < 0.22:
                    has_s = True
        return has_cl_br, has_s

    def score_components(
        self,
        neutral_mass: float,
        pooled_nls: np.ndarray,
        pooled_ints: np.ndarray,
        raw_mzs: Optional[np.ndarray] = None,
        raw_ints: Optional[np.ndarray] = None,
        fg_71_pred: Optional[np.ndarray] = None,
        ppm_window: float = W("nb4.priors.ppm_window"),
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Candidate formulas within +/- ppm_window and their four component scores (before weighting)."""
        formulas, exact_masses, atom_mat, rdbe_vec, hetero_log1p = (
            self.bank.query_mass_window(neutral_mass, ppm_tol=ppm_window)
        )
        n_cand = len(formulas)
        zeros = {k: np.zeros(n_cand, dtype=np.float64) for k in self.SCORE_KEYS}
        if n_cand == 0:
            return formulas, zeros

        # 1. Precursor Mass PPM Score (Gaussian log-likelihood in [0, 1])
        ppm_errors = np.abs(exact_masses - neutral_mass) / neutral_mass * 1e6
        score_mass = np.exp(-0.5 * (ppm_errors / self.ppm_sigma) ** 2)

        # 2. Diagnostic Neutral Loss Subformula Score (SIRIUS / MIST-CF style)
        score_nl = np.zeros(n_cand, dtype=np.float64)
        valid_nl_mask = (pooled_nls > 1.5) & (pooled_ints > 0.05)
        obs_nls = pooled_nls[valid_nl_mask]
        obs_weights = pooled_ints[valid_nl_mask]
        nl_hit, nl_miss = W("nb4.nl.explained"), W("nb4.nl.unexplained")

        if len(obs_nls) > 0:
            # Match observed neutral losses against pre-compiled diagnostic neutral loss table
            diff_matrix = np.abs(obs_nls[:, None] - NL_EXACT_MASSES[None, :])  # (N_obs, N_diag)
            obs_idx, diag_idx = np.where(diff_matrix <= self.nl_tol_da)

            for o_i, d_i in zip(obs_idx, diag_idx):
                req_atoms = NL_ATOM_MATRIX[d_i]  # (10,)
                w_peak = float(obs_weights[o_i])
                # Candidate formula can explain this neutral loss iff atom_mat >= req_atoms for all 10 elements
                can_form_loss = np.all(atom_mat >= req_atoms[None, :], axis=1)
                score_nl += np.where(can_form_loss, nl_hit * w_peak, nl_miss * w_peak)

        # 3. Raw MS2 Isotope Pattern Score (Cl/Br and S M+2 signatures; raw_ints must be linear intensities)
        score_iso = np.zeros(n_cand, dtype=np.float64)
        if raw_mzs is not None and raw_ints is not None:
            has_cl_br, has_s = self.detect_halogen_and_sulfur_isotopes(raw_mzs, raw_ints)
            cand_has_cl_br = (atom_mat[:, 7] + atom_mat[:, 8]) > 0
            cand_has_s = atom_mat[:, 5] > 0
            if has_cl_br:
                score_iso += np.where(cand_has_cl_br, W("nb4.iso.clbr_detected_match"), W("nb4.iso.clbr_detected_mismatch"))
            else:
                score_iso += np.where(cand_has_cl_br, W("nb4.iso.clbr_absent_but_present"), W("nb4.iso.clbr_absent_and_absent"))
            if has_s:
                score_iso += np.where(cand_has_s, W("nb4.iso.s_detected_match"), W("nb4.iso.s_detected_mismatch"))

        # 4. Mass2SMILES 71-D Neural Heteroatom + Ring RDBE Prior Score
        score_neural = np.zeros(n_cand, dtype=np.float64)
        if fg_71_pred is not None and len(fg_71_pred) >= 71:
            # Indices 60..67: predicted log1p counts for [N, O, S, P, F, Cl, Br, I]
            pred_hetero = np.asarray(fg_71_pred[60:68], dtype=np.float32)
            # Weight N, O, S, P, Cl, Br agreement strongly
            elem_weights = np.array(W.vec("nb4.neural.elem_weight", HETEROATOM_SYMBOLS), dtype=np.float32)
            l1_hetero = np.sum(
                np.abs(hetero_log1p - pred_hetero[None, :]) * elem_weights[None, :],
                axis=1,
            ) / np.sum(elem_weights)
            score_neural += np.exp(-W("nb4.neural.decay") * l1_hetero)

            # Indices 69..70: log1p(aromatic rings) and log1p(aliphatic rings)
            est_arom_rings = max(0.0, float(np.expm1(fg_71_pred[69])))
            est_aliph_rings = max(0.0, float(np.expm1(fg_71_pred[70])))
            min_expected_rdbe = (W("nb4.neural.rdbe_per_aromatic_ring") * est_arom_rings
                                 + W("nb4.neural.rdbe_per_aliphatic_ring") * est_aliph_rings)
            # Penalize formulas whose RDBE is far below the neural ring lower bound
            rdbe_deficit = np.clip(min_expected_rdbe - rdbe_vec, 0.0, None)
            score_neural -= W("nb4.neural.rdbe_deficit_penalty") * rdbe_deficit

        return formulas, {"mass": score_mass, "nl": score_nl, "iso": score_iso, "neural": score_neural}

    def combine(self, components: Dict[str, np.ndarray], score_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Total composite formula log-posterior score."""
        w = score_weights or self.score_weights
        return sum(w[k] * components[k] for k in self.SCORE_KEYS)

    def score_and_select_top_k_formulas(
        self,
        neutral_mass: float,
        pooled_nls: np.ndarray,
        pooled_ints: np.ndarray,
        raw_mzs: Optional[np.ndarray] = None,
        raw_ints: Optional[np.ndarray] = None,
        fg_71_pred: Optional[np.ndarray] = None,
        ppm_window: float = W("nb4.priors.ppm_window"),
        top_k: int = W.int("nb4.priors.top_k_formulas"),
    ) -> List[str]:
        """
        Scores all valid candidate formulas within +/- ppm_window and returns the top-K
        Hill-notation formula strings for OfflineMassAndFormulaRetriever.
        """
        formulas, components = self.score_components(
            neutral_mass, pooled_nls, pooled_ints, raw_mzs, raw_ints, fg_71_pred, ppm_window
        )
        if len(formulas) <= 1:
            return [str(f) for f in formulas]
        ranked_idx = np.argsort(-self.combine(components))[:top_k]
        return [str(formulas[i]) for i in ranked_idx]

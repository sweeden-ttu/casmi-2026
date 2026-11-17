# [Colab Formula Filter Cell 2] Mass-Sorted Formula Table + De Novo CHNOPS Validator
class CandidateFormulaBank:
    """
    Stores a mass-sorted table of unique molecular formulas, their exact monoisotopic
    masses, 10-element stoichiometry matrices, and RDBE values for O(log N) slicing.
    """

    def __init__(self, catalog_formulas: List[str]):
        unique_formulas = sorted(set(catalog_formulas))
        valid_f, vecs = [], []
        for f_str in unique_formulas:
            v = parse_formula_to_vector(str(f_str))
            if v is not None and self.passes_seniors_and_golden_rules(v):
                valid_f.append(str(f_str))
                vecs.append(v)

        if vecs:
            mat = np.stack(vecs, axis=0)
        else:
            mat = np.zeros((0, len(ELEMENTS)), dtype=np.int16)

        masses = mat.astype(np.float64) @ EXACT_ATOMIC_MASSES
        order = np.argsort(masses)

        self.formulas = np.array([valid_f[i] for i in order], dtype=object)
        self.atom_matrix = mat[order]                             # (M, 10) int16
        self.exact_masses = masses[order]                         # (M,) float64
        self.rdbe = (self.atom_matrix @ RDBE_COEFFS) + 1.0        # (M,) float64
        # Precompute log1p heteroatom counts [N, O, S, P, F, Cl, Br, I] to match Mass2SMILES indices 60..67
        # Note order in Mass2SMILES: ["N", "O", "S", "P", "F", "Cl", "Br", "I"] -> indices [2, 3, 5, 4, 6, 7, 8, 9]
        m2s_cols = [2, 3, 5, 4, 6, 7, 8, 9]
        self.hetero_log1p = np.log1p(
            self.atom_matrix[:, m2s_cols].astype(np.float32)
        )  # (M, 8)

    @staticmethod
    def passes_seniors_and_golden_rules(vec: np.ndarray) -> bool:
        """
        Enforces Senior's Valence Theorem (Nitrogen Rule / integer RDBE >= -0.5)
        and Kind & Fiehn (2007) H/C and heteroatom/carbon bounds for organic molecules.
        """
        c, h = int(vec[0]), int(vec[1])
        if c <= 0 or h <= 0:
            return False
        # Double-check RDBE is an exact integer >= 0.0 (sum of odd-valence atoms must be even)
        two_rdbe = 2 * c - (h + int(vec[6]) + int(vec[7]) + int(vec[8]) + int(vec[9])) + (int(vec[2]) + int(vec[4])) + 2
        if two_rdbe < 0 or (two_rdbe % 2 != 0):
            return False
        # Seven Golden Rules H/C ratio bounds (0.2 <= H/C <= 3.5) and O/C <= 1.5, N/C <= 1.3
        if not (0.15 * c <= h <= 3.5 * c + 2):
            return False
        if int(vec[3]) > int(1.5 * c + 2) or int(vec[2]) > int(1.3 * c + 2):
            return False
        return True

    def query_mass_window(
        self, neutral_mass: float, ppm_tol: float = 15.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Slices candidate formulas within [neutral_mass - tol, neutral_mass + tol] in O(log M).
        If fewer than 3 catalog formulas exist in the window, enumerates de novo CHNOPS formulas.
        """
        tol_da = max(neutral_mass * (ppm_tol * 1e-6), 0.012)
        lo = int(np.searchsorted(self.exact_masses, neutral_mass - tol_da, side="left"))
        hi = int(np.searchsorted(self.exact_masses, neutral_mass + tol_da, side="right"))

        if hi - lo >= 3:
            return (
                self.formulas[lo:hi],
                self.exact_masses[lo:hi],
                self.atom_matrix[lo:hi],
                self.rdbe[lo:hi],
                self.hetero_log1p[lo:hi],
            )

        # Fallback: fast de novo CHNO(S) enumeration within tolerance window
        return self._enumerate_denovo_formulas(neutral_mass, tol_da)

    def _enumerate_denovo_formulas(
        self, neutral_mass: float, tol_da: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c_min = max(1, int((neutral_mass - 180.0) / 14.0))
        c_max = min(85, int(neutral_mass / 12.0))

        found_f, found_m, found_v, found_r, found_h = [], [], [], [], []
        m2s_cols = [2, 3, 5, 4, 6, 7, 8, 9]

        for c in range(c_min, c_max + 1):
            rem_c = neutral_mass - c * EXACT_ATOMIC_MASSES[0]
            if rem_c < 1.0:
                continue
            for n in range(0, min(12, int(rem_c / 14.0) + 1)):
                rem_cn = rem_c - n * EXACT_ATOMIC_MASSES[2]
                for o in range(0, min(20, int(rem_cn / 15.99) + 1)):
                    rem_cno = rem_cn - o * EXACT_ATOMIC_MASSES[3]
                    for s in range(0, min(3, int(rem_cno / 31.97) + 1)):
                        rem_h = rem_cno - s * EXACT_ATOMIC_MASSES[5]
                        if rem_h < 0.5:
                            continue
                        h = int(round(rem_h / EXACT_ATOMIC_MASSES[1]))
                        vec = np.array([c, h, n, o, 0, s, 0, 0, 0, 0], dtype=np.int16)
                        if not self.passes_seniors_and_golden_rules(vec):
                            continue
                        exact_m = float(vec.astype(np.float64) @ EXACT_ATOMIC_MASSES)
                        if abs(exact_m - neutral_mass) <= tol_da:
                            found_f.append(vector_to_hill_formula(vec))
                            found_m.append(exact_m)
                            found_v.append(vec)
                            found_r.append(float((vec @ RDBE_COEFFS) + 1.0))
                            found_h.append(np.log1p(vec[m2s_cols].astype(np.float32)))

        if not found_f:
            return (
                np.empty(0, dtype=object),
                np.empty(0, dtype=np.float64),
                np.empty((0, 10), dtype=np.int16),
                np.empty(0, dtype=np.float64),
                np.empty((0, 8), dtype=np.float32),
            )
        return (
            np.array(found_f, dtype=object),
            np.array(found_m, dtype=np.float64),
            np.stack(found_v, axis=0),
            np.array(found_r, dtype=np.float64),
            np.stack(found_h, axis=0),
        )
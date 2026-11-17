"""
Multi-Modal Spectral-to-Structure Modeling & Reranking Architecture for CASMI 2026:

1. `SpectralLibraryIndex`:
   Indexes training spectra (`train.parquet`) by neutral precursor mass and computes:
   - Exact Spectral Entropy Similarity (Li et al., Nature Methods)
   - Binned m/z + Neutral Loss Cosine Similarity
   - Structural Network Propagation (Tanimoto consensus between a candidate isomer and
     the top-K closest spectral neighbors in `train.parquet`).

2. `SpectrumToFingerprintModel`:
   - In PyTorch GPU mode (Colab): Trains a residual MLP / Peak-Set Transformer (`FPNet`)
     with Asymmetric Focal / BCE loss to predict 2215-bit molecular fingerprints (2048 Morgan + 167 MACCS)
     from binned spectra + precursor metadata.
   - In CPU / fallback mode: Fits a fast multi-output Ridge / PLS projection from spectral vectors
     to molecular fingerprints.

3. `MultiSpectrumReranker`:
   Aggregates features across all 1-16 `spectrum_id`s belonging to each `molecule_id`,
   trains a gradient-boosted ranker (LightGBM `lambdarank` if installed, or Scikit-Learn
   `HistGradientBoostingClassifier` fallback), and outputs the deduplicated Top-25 SMILES list.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge

from casmi26.chemistry import (
    apply_electron_mass_correction,
    deduplicate_ranked_smiles,
    precursor_mz_to_neutral_mass,
    triangulate_molecule_neutral_mass,
)
from casmi26.config import TOP_K_CANDIDATES
from casmi26.features import (
    FINGERPRINT_DIM,
    calibrated_fp_log_likelihood,
    clean_and_normalize_spectrum,
    continuous_tanimoto_similarity,
    exact_spectral_entropy_similarity,
    smiles_to_fingerprint,
    spectrum_to_binned_vector,
)
from casmi26.retrieval import CandidatePool
from casmi26.fusion import linear_blend
from casmi26.weights import analytical_weights

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# 1. Reference Spectral Library Index (Class 1 Direct Hits + Class 2 Analog Network)
# ---------------------------------------------------------------------------

@dataclass
class SpectralLibraryIndex:
    """Sorted index of reference MS/MS spectra for fast mass-window + analog lookup."""
    neutral_masses: np.ndarray        # float64 sorted, shape (S,)
    inchikey14s: np.ndarray           # object array, shape (S,)
    smiles: np.ndarray                # object array, shape (S,)
    binned_vectors: np.ndarray        # float32 L2-normalized, shape (S, 1536)
    cleaned_mzs: List[np.ndarray]     # list of float32 arrays
    cleaned_probs: List[np.ndarray]   # list of float32 arrays

    @classmethod
    def build(
        cls,
        train_df: pd.DataFrame,
        apply_enveda180_electron_fix: bool = True,
    ) -> "SpectralLibraryIndex":
        n = len(train_df)
        if n == 0:
            return cls(
                neutral_masses=np.empty(0, dtype=np.float64),
                inchikey14s=np.empty(0, dtype=object),
                smiles=np.empty(0, dtype=object),
                binned_vectors=np.empty((0, 1536), dtype=np.float32),
                cleaned_mzs=[],
                cleaned_probs=[],
            )

        neutral_masses = np.zeros(n, dtype=np.float64)
        inchikey14s = np.empty(n, dtype=object)
        smiles = np.empty(n, dtype=object)
        binned_vectors = np.zeros((n, 1536), dtype=np.float32)
        cleaned_mzs: List[np.ndarray] = []
        cleaned_probs: List[np.ndarray] = []

        for idx, row in enumerate(train_df.itertuples(index=False)):
            mz = float(getattr(row, "precursor_mz", 0.0))
            add = str(getattr(row, "adduct", "[M+H]+"))
            mode = str(getattr(row, "ionization_mode", "positive"))
            ingest = str(getattr(row, "ingest_lib", "")).lower()

            mzs_raw = getattr(row, "ms2_mzs", [])
            ints_raw = getattr(row, "ms2_normalized_intensities", [])

            if apply_enveda180_electron_fix and "enveda-180" in ingest and mode.startswith("pos"):
                mz = float(apply_electron_mass_correction(mz, mode, direction="subtract"))
                mzs_raw = apply_electron_mass_correction(np.asarray(mzs_raw, dtype=np.float32), mode, direction="subtract")

            nm = precursor_mz_to_neutral_mass(mz, add, mode)
            neutral_masses[idx] = nm
            inchikey14s[idx] = str(getattr(row, "inchikey14", ""))
            smiles[idx] = str(getattr(row, "normalized_smiles", ""))

            c_mzs, c_probs = clean_and_normalize_spectrum(mzs_raw, ints_raw, mz)
            cleaned_mzs.append(c_mzs)
            cleaned_probs.append(c_probs)
            binned_vectors[idx] = spectrum_to_binned_vector(c_mzs, c_probs, mz)

        order = np.argsort(neutral_masses)
        return cls(
            neutral_masses=neutral_masses[order],
            inchikey14s=inchikey14s[order],
            smiles=smiles[order],
            binned_vectors=binned_vectors[order],
            cleaned_mzs=[cleaned_mzs[i] for i in order],
            cleaned_probs=[cleaned_probs[i] for i in order],
        )

    def query_spectral_matches(
        self,
        neutral_mass: float,
        query_binned_vec: np.ndarray,
        query_c_mzs: np.ndarray,
        query_c_probs: np.ndarray,
        ppm_window: float = 20.0,
        analog_mass_window_da: float = 50.0,
        top_k_analogs: int = 15,
    ) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, float]]]:
        """
        Returns:
        1. `exact_mass_matches`: Dict mapping `inchikey14` -> `(max_entropy_sim, max_cosine_sim)`
           among reference spectra within `ppm_window` of `neutral_mass`.
        2. `top_analogs`: List of `(smiles, cosine_sim)` for the closest reference spectra within
           `+/- analog_mass_window_da` (used for Network Reranking of Class 2 isomers).
        """
        if self.neutral_masses.size == 0:
            return {}, []

        # 1. Exact neutral mass library matching
        tol_da = max(neutral_mass * ppm_window * 1e-6, 0.01)
        l_idx = int(np.searchsorted(self.neutral_masses, neutral_mass - tol_da, side="left"))
        r_idx = int(np.searchsorted(self.neutral_masses, neutral_mass + tol_da, side="right"))

        exact_matches: Dict[str, Tuple[float, float]] = {}
        if r_idx > l_idx:
            cos_sims = self.binned_vectors[l_idx:r_idx] @ query_binned_vec
            for offset, cos_val in enumerate(cos_sims):
                lib_idx = l_idx + offset
                ik14 = self.inchikey14s[lib_idx]
                ent_val = exact_spectral_entropy_similarity(
                    query_c_mzs,
                    query_c_probs,
                    self.cleaned_mzs[lib_idx],
                    self.cleaned_probs[lib_idx],
                )
                prev_ent, prev_cos = exact_matches.get(ik14, (0.0, 0.0))
                exact_matches[ik14] = (max(prev_ent, ent_val), max(prev_cos, float(cos_val)))

        # 2. Broad mass window analog search for Network Reranking
        al_left = int(np.searchsorted(self.neutral_masses, neutral_mass - analog_mass_window_da, side="left"))
        al_right = int(np.searchsorted(self.neutral_masses, neutral_mass + analog_mass_window_da, side="right"))
        top_analogs: List[Tuple[str, float]] = []
        if al_right > al_left:
            # Subsample if huge
            span = al_right - al_left
            if span > 4096:
                step = max(1, span // 4096)
                cand_indices = np.arange(al_left, al_right, step)
            else:
                cand_indices = np.arange(al_left, al_right)

            analog_cos = self.binned_vectors[cand_indices] @ query_binned_vec
            k = min(top_k_analogs, analog_cos.size)
            if k > 0:
                best_local = np.argpartition(analog_cos, -k)[-k:]
                for loc in best_local:
                    sim_val = float(analog_cos[loc])
                    if sim_val > 0.05:
                        top_analogs.append((str(self.smiles[cand_indices[loc]]), sim_val))

        return exact_matches, top_analogs


# ---------------------------------------------------------------------------
# 2. Spectrum-to-Fingerprint Predictor (`FPNet` PyTorch GPU + Ridge CPU Fallback)
# ---------------------------------------------------------------------------

if _HAS_TORCH:
    class AsymmetricFingerprintLoss(nn.Module):
        """
        Asymmetric Focal + Positive-Weighted Binary Cross-Entropy Loss for sparse
        2215-bit molecular fingerprint prediction (Probe 2C).
        """
        def __init__(self, pos_weight_val: float = 4.0, gamma_neg: float = 2.0):
            super().__init__()
            self.pos_weight_val = pos_weight_val
            self.gamma_neg = gamma_neg

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            probs = torch.sigmoid(logits)
            pos_loss = -targets * torch.log(probs.clamp(min=1e-6)) * self.pos_weight_val
            neg_mod = (probs ** self.gamma_neg)
            neg_loss = -(1.0 - targets) * neg_mod * torch.log((1.0 - probs).clamp(min=1e-6))
            return torch.mean(pos_loss + neg_loss)


    class HardIsomerContrastiveLoss(nn.Module):
        """
        Cross-modal margin + InfoNCE loss that forces predicted spectrum fingerprints/embeddings
        to have higher similarity to the true structure than to same-mass / same-formula
        hard structural isomers sampled from COCONUT 2.0 and PubChem (Probe 2A & 2B).
        """
        def __init__(self, hard_isomer_margin: float = 0.15, temperature: float = 0.07):
            super().__init__()
            self.hard_isomer_margin = hard_isomer_margin
            self.temperature = temperature
            self.margin_loss = nn.MarginRankingLoss(margin=hard_isomer_margin)

        def forward(
            self,
            pred_fp_probs: torch.Tensor,
            true_fps: torch.Tensor,
            hard_neg_fps: torch.Tensor,
        ) -> torch.Tensor:
            # pred_fp_probs: (B, D), true_fps: (B, D), hard_neg_fps: (B, K, D)
            p_norm = torch.nn.functional.normalize(pred_fp_probs, dim=-1)
            t_norm = torch.nn.functional.normalize(true_fps, dim=-1)
            pos_sim = torch.sum(p_norm * t_norm, dim=-1, keepdim=True)  # (B, 1)

            if hard_neg_fps.ndim == 2:
                hard_neg_fps = hard_neg_fps.unsqueeze(1)
            n_norm = torch.nn.functional.normalize(hard_neg_fps, dim=-1)  # (B, K, D)
            neg_sims = torch.bmm(n_norm, p_norm.unsqueeze(-1)).squeeze(-1)  # (B, K)

            # InfoNCE over [true_isomer, hard_isomer_1, ..., hard_isomer_K]
            logits = torch.cat([pos_sim, neg_sims], dim=1) / self.temperature
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            infonce = torch.nn.functional.cross_entropy(logits, labels)

            # Hardest-isomer margin ranking loss
            hardest_neg_sim, _ = torch.max(neg_sims, dim=1)
            target_ones = torch.ones_like(hardest_neg_sim)
            margin_term = self.margin_loss(pos_sim.squeeze(-1), hardest_neg_sim, target_ones)
            return infonce + margin_term


    class FPNetTorch(nn.Module):
        """
        Residual MLP + Gated Linear Unit architecture mapping 1536-dim binned spectrum
        (m/z + neutral losses) + precursor mass metadata to 2215-bit molecular fingerprints.
        Used when training on Google Colab GPUs.
        """
        def __init__(self, in_dim: int = 1538, hidden_dim: int = 1024, out_dim: int = FINGERPRINT_DIM):
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.15),
            )
            self.block1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.block2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.act = nn.SiLU()
            self.head = nn.Linear(hidden_dim, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.input_layer(x)
            h = self.act(h + self.block1(h))
            h = self.act(h + self.block2(h))
            return self.head(h)


class SpectrumToFingerprintModel:
    """
    Unified wrapper that trains either `FPNetTorch` on GPU (Colab) with
    `AsymmetricFingerprintLoss` + `HardIsomerContrastiveLoss` against same-mass COCONUT/PubChem
    isomers, or a fast vectorized Ridge projection with hard-negative isomer augmentation on CPU.
    """
    def __init__(self, use_torch_if_available: bool = True, ridge_alpha: float = 10.0):
        self.use_torch = bool(use_torch_if_available and _HAS_TORCH and torch.cuda.is_available())
        self.ridge_alpha = ridge_alpha
        self.ridge_model: Optional[Ridge] = None
        self.torch_model: Optional[object] = None
        self.device: str = "cuda" if self.use_torch else "cpu"

    @staticmethod
    def _build_input_matrix(binned_spectra: np.ndarray, neutral_masses: np.ndarray, is_pos_mode: np.ndarray) -> np.ndarray:
        mass_feat = (neutral_masses[:, None].astype(np.float32) / 1000.0)
        mode_feat = is_pos_mode[:, None].astype(np.float32)
        return np.hstack([binned_spectra, mass_feat, mode_feat])

    def fit(
        self,
        binned_spectra: np.ndarray,
        neutral_masses: np.ndarray,
        is_pos_mode: np.ndarray,
        target_fps: np.ndarray,
        hard_neg_fps: Optional[np.ndarray] = None,
        epochs: int = 12,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> "SpectrumToFingerprintModel":
        X = self._build_input_matrix(binned_spectra, neutral_masses, is_pos_mode)

        if self.use_torch and _HAS_TORCH:
            model = FPNetTorch(in_dim=X.shape[1], hidden_dim=1024, out_dim=target_fps.shape[1]).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            fp_criterion = AsymmetricFingerprintLoss(pos_weight_val=4.0, gamma_neg=2.0)
            contrastive_criterion = HardIsomerContrastiveLoss(hard_isomer_margin=0.15)

            if hard_neg_fps is not None and len(hard_neg_fps) == len(X):
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X),
                    torch.from_numpy(target_fps.astype(np.float32)),
                    torch.from_numpy(hard_neg_fps.astype(np.float32)),
                )
            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X),
                    torch.from_numpy(target_fps.astype(np.float32)),
                )

            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.train()
            for _ in range(epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    bx = batch[0].to(self.device)
                    by = batch[1].to(self.device)
                    logits = model(bx)
                    loss = fp_criterion(logits, by)
                    if len(batch) == 3:
                        b_neg = batch[2].to(self.device)
                        loss = loss + 0.35 * contrastive_criterion(torch.sigmoid(logits), by, b_neg)
                    loss.backward()
                    optimizer.step()
            model.eval()
            self.torch_model = model
        else:
            # On CPU, if hard same-mass isomer negatives are supplied, sharpen target contrast
            # against isomer-shared common bits so discriminative substructure bits receive higher weight
            y_target = target_fps.astype(np.float32)
            if hard_neg_fps is not None and len(hard_neg_fps) == len(X):
                neg_mean = np.mean(hard_neg_fps, axis=1) if hard_neg_fps.ndim == 3 else hard_neg_fps
                y_target = np.clip(y_target - 0.15 * (neg_mean * (1.0 - y_target)), 0.0, 1.0)
            self.ridge_model = Ridge(alpha=self.ridge_alpha)
            self.ridge_model.fit(X, y_target)

        return self

    def predict_proba(
        self,
        binned_spectra: np.ndarray,
        neutral_masses: np.ndarray,
        is_pos_mode: np.ndarray,
    ) -> np.ndarray:
        X = self._build_input_matrix(binned_spectra, neutral_masses, is_pos_mode)
        if self.torch_model is not None and _HAS_TORCH:
            self.torch_model.eval()
            with torch.no_grad():
                t_x = torch.from_numpy(X).to(self.device)
                probs = torch.sigmoid(self.torch_model(t_x)).cpu().numpy()
            return np.clip(probs, 1e-4, 1.0 - 1e-4).astype(np.float32)

        if self.ridge_model is not None:
            raw = self.ridge_model.predict(X)
            return np.clip(raw, 1e-4, 1.0 - 1e-4).astype(np.float32)

        return np.full((len(X), FINGERPRINT_DIM), 0.05, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. End-to-End Multi-Spectrum Candidate Scorer & GBDT Learned Reranker
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "lib_max_entropy_sim",
    "lib_mean_entropy_sim",
    "lib_max_cosine_sim",
    "fp_tanimoto_sim",
    "fp_log_likelihood",
    "fp_cosine_sim",
    "network_max_tanimoto",
    "network_weighted_tanimoto",
    "source_prior",
    "np_likeness_score",
    "mass_error_da",
    "mass_error_ppm",
    "num_spectra_for_mol",
    "neutral_mass_std_da",
]


class CASMIPipeline:
    """
    End-to-end retrieval + multi-spectrum feature extraction + learned GBDT reranker
    for Enveda CASMI 2026.

    Can be trained in Google Colab, serialized to `kaggle_dataset_export/`, uploaded as a
    Kaggle Dataset, and executed offline in a Kaggle submission notebook.
    """

    def __init__(
        self,
        candidate_pool: CandidatePool,
        spectral_index: SpectralLibraryIndex,
        fp_model: SpectrumToFingerprintModel,
        ppm_tolerance: float = 15.0,
        max_candidates_per_mol: int = 300,
    ):
        self.candidate_pool = candidate_pool
        self.spectral_index = spectral_index
        self.fp_model = fp_model
        self.ppm_tolerance = ppm_tolerance
        self.max_candidates_per_mol = max_candidates_per_mol
        self.reranker: Optional[object] = None
        self._fp_cache: Dict[str, np.ndarray] = {}

    def _get_smiles_fp(self, smiles: str) -> np.ndarray:
        fp = self._fp_cache.get(smiles)
        if fp is None:
            fp = smiles_to_fingerprint(smiles)
            if len(self._fp_cache) < 250_000:
                self._fp_cache[smiles] = fp
        return fp

    def build_molecule_candidate_features(
        self,
        mol_spectra_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Given all rows (`spectrum_id`s) for a single `molecule_id`, triangulates the
        consensus neutral mass, retrieves candidates from `CandidatePool`, aggregates
        spectral library, neural fingerprint, and network rerank signals across all
        spectra of that molecule, and returns a feature DataFrame.
        """
        mzs = mol_spectra_df["precursor_mz"].to_numpy(dtype=np.float64)
        adducts = mol_spectra_df["adduct"].astype(str).tolist() if "adduct" in mol_spectra_df.columns else ["[M+H]+"] * len(mzs)
        modes = mol_spectra_df["ionization_mode"].astype(str).tolist() if "ionization_mode" in mol_spectra_df.columns else ["positive"] * len(mzs)

        consensus_mass, mass_std_da = triangulate_molecule_neutral_mass(mzs, adducts, modes)
        cands_df = self.candidate_pool.query_by_neutral_mass(
            consensus_mass,
            ppm_tolerance=self.ppm_tolerance,
            max_candidates=self.max_candidates_per_mol,
        )
        if cands_df.empty:
            return cands_df

        n_spec = len(mol_spectra_df)
        spec_binned = np.zeros((n_spec, 1536), dtype=np.float32)
        spec_nm = np.zeros(n_spec, dtype=np.float32)
        spec_pos = np.zeros(n_spec, dtype=np.float32)

        per_ik_ent: Dict[str, List[float]] = {}
        per_ik_cos: Dict[str, List[float]] = {}
        analog_pool: List[Tuple[str, float]] = []

        for s_idx, row in enumerate(mol_spectra_df.itertuples(index=False)):
            p_mz = float(getattr(row, "precursor_mz", 0.0))
            add = str(getattr(row, "adduct", "[M+H]+"))
            mode = str(getattr(row, "ionization_mode", "positive"))
            nm = precursor_mz_to_neutral_mass(p_mz, add, mode)

            raw_mzs = getattr(row, "ms2_mzs", [])
            raw_ints = getattr(row, "ms2_normalized_intensities", [])
            c_mzs, c_probs = clean_and_normalize_spectrum(raw_mzs, raw_ints, p_mz)
            b_vec = spectrum_to_binned_vector(c_mzs, c_probs, p_mz)

            spec_binned[s_idx] = b_vec
            spec_nm[s_idx] = float(nm)
            spec_pos[s_idx] = 1.0 if mode.lower().startswith("pos") else 0.0

            exact_hits, analogs = self.spectral_index.query_spectral_matches(
                nm, b_vec, c_mzs, c_probs, ppm_window=self.ppm_tolerance
            )
            for ik14, (ent_s, cos_s) in exact_hits.items():
                per_ik_ent.setdefault(ik14, []).append(ent_s)
                per_ik_cos.setdefault(ik14, []).append(cos_s)
            analog_pool.extend(analogs)

        # Predict molecular fingerprints for each spectrum and average across the molecule_id
        pred_fps_per_spec = self.fp_model.predict_proba(spec_binned, spec_nm, spec_pos)
        consensus_pred_fp = np.mean(pred_fps_per_spec, axis=0)

        # Compute candidate fingerprint matrix
        cand_smiles = cands_df["smiles"].tolist()
        cand_ik14s = cands_df["inchikey14"].tolist()
        cand_fps = np.vstack([self._get_smiles_fp(s) for s in cand_smiles])

        fp_tanimoto = continuous_tanimoto_similarity(consensus_pred_fp, cand_fps)
        fp_ll = calibrated_fp_log_likelihood(consensus_pred_fp, cand_fps)
        cand_norms = np.linalg.norm(cand_fps, axis=1) * float(np.linalg.norm(consensus_pred_fp)) + 1e-8
        fp_cosine = (cand_fps @ consensus_pred_fp) / cand_norms

        # Spectral Network Rerank features (similarity to top spectral neighbors in train.parquet)
        net_max_tan = np.zeros(len(cands_df), dtype=np.float32)
        net_wt_tan = np.zeros(len(cands_df), dtype=np.float32)
        if analog_pool:
            # Keep top 15 unique analog structures
            analog_pool.sort(key=lambda x: x[1], reverse=True)
            seen_a: set[str] = set()
            dedup_analogs: List[Tuple[str, float]] = []
            for a_smi, a_sim in analog_pool:
                if a_smi not in seen_a:
                    seen_a.add(a_smi)
                    dedup_analogs.append((a_smi, a_sim))
                if len(dedup_analogs) >= 15:
                    break

            analog_fps = np.vstack([self._get_smiles_fp(a_smi) for a_smi, _ in dedup_analogs])
            analog_weights = np.array([a_sim for _, a_sim in dedup_analogs], dtype=np.float32)
            analog_weights /= np.sum(analog_weights) + 1e-8

            # Pairwise Tanimoto (M_cands, K_analogs)
            dot_ca = cand_fps @ analog_fps.T
            c_bits = np.sum(cand_fps, axis=1, keepdims=True)
            a_bits = np.sum(analog_fps, axis=1, keepdims=True).T
            tan_matrix = dot_ca / (c_bits + a_bits - dot_ca + 1e-8)

            net_max_tan = np.max(tan_matrix, axis=1).astype(np.float32)
            net_wt_tan = (tan_matrix @ analog_weights).astype(np.float32)

        lib_max_ent = np.array([max(per_ik_ent.get(ik, [0.0])) for ik in cand_ik14s], dtype=np.float32)
        lib_mean_ent = np.array(
            [sum(per_ik_ent.get(ik, [0.0])) / max(1, n_spec) for ik in cand_ik14s], dtype=np.float32
        )
        lib_max_cos = np.array([max(per_ik_cos.get(ik, [0.0])) for ik in cand_ik14s], dtype=np.float32)

        cands_df = cands_df.copy()
        cands_df["lib_max_entropy_sim"] = lib_max_ent
        cands_df["lib_mean_entropy_sim"] = lib_mean_ent
        cands_df["lib_max_cosine_sim"] = lib_max_cos
        cands_df["fp_tanimoto_sim"] = fp_tanimoto
        cands_df["fp_log_likelihood"] = fp_ll
        cands_df["fp_cosine_sim"] = fp_cosine
        cands_df["network_max_tanimoto"] = net_max_tan
        cands_df["network_weighted_tanimoto"] = net_wt_tan
        cands_df["num_spectra_for_mol"] = float(n_spec)
        cands_df["neutral_mass_std_da"] = float(mass_std_da)

        return cands_df

    def fit_reranker(
        self,
        val_queries_df: pd.DataFrame,
        val_truth_df: pd.DataFrame,
    ) -> "CASMIPipeline":
        """
        Builds candidate feature tables across a labeled query panel and trains a GBDT
        reranker (LightGBM `lambdarank` if installed, else HistGradientBoostingClassifier).
        """
        truth_map = dict(zip(val_truth_df["molecule_id"], val_truth_df["inchikey14"]))
        feature_tables: List[pd.DataFrame] = []
        group_sizes: List[int] = []

        for mol_id, grp in val_queries_df.groupby("molecule_id", sort=False):
            true_ik14 = truth_map.get(mol_id)
            if not true_ik14:
                continue
            cand_feats = self.build_molecule_candidate_features(grp)
            if cand_feats.empty:
                continue
            cand_feats["label"] = (cand_feats["inchikey14"] == true_ik14).astype(np.int32)
            feature_tables.append(cand_feats)
            group_sizes.append(len(cand_feats))

        if not feature_tables:
            return self

        full_df = pd.concat(feature_tables, ignore_index=True)
        X = full_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = full_df["label"].to_numpy(dtype=np.int32)

        if _HAS_LIGHTGBM and np.sum(y) >= 5:
            ranker = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                ndcg_eval_at=[1, 5, 10, 25],
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                subsample_for_bin=50000,
                random_state=42,
                verbose=-1,
            )
            ranker.fit(X, y, group=group_sizes)
            self.reranker = ranker
        elif len(np.unique(y)) > 1:
            clf = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42,
            )
            clf.fit(X, y)
            self.reranker = clf

        return self

    def score_candidates(
        self,
        cand_feats: pd.DataFrame,
        weights: Dict[str, float] | None = None,
    ) -> np.ndarray:
        """Scores candidates using the learned GBDT reranker + the Evidently blend.

        Always returns float64. The no-reranker path matches ``apply_blend``.

        ``weights`` defaults to ``experiments/weights/current.json``. The GBDT
        mix (0.65 / 0.35) is unchanged; that file's importances are empty, so
        there is no learned ranker coefficient to load.
        """
        blend = analytical_weights() if weights is None else weights
        linear_score, _used = linear_blend(cand_feats, blend)

        if self.reranker is None:
            return np.asarray(linear_score, dtype=np.float64)

        X = cand_feats[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if hasattr(self.reranker, "predict_proba"):
            gbdt_score = self.reranker.predict_proba(X)[:, 1]
        else:
            gbdt_score = self.reranker.predict(X)

        # Same float64 contract as the analytical path and apply_blend.
        mixed = 0.65 * np.asarray(gbdt_score, dtype=np.float64) + 0.35 * linear_score
        return np.asarray(mixed, dtype=np.float64)

    def predict_top25_for_test_df(
        self,
        test_df: pd.DataFrame,
        max_k: int = TOP_K_CANDIDATES,
    ) -> pd.DataFrame:
        """
        Generates the official `submission.csv` DataFrame (`molecule_id`, `smiles`)
        where `smiles` is a semicolon-separated string of up to 25 `inchikey14`-deduplicated
        canonical SMILES strings per `molecule_id`.
        """
        rows = []
        fallback_smiles = self.candidate_pool.smiles[:max_k].tolist() if len(self.candidate_pool) > 0 else ["C"]

        for mol_id, grp in test_df.groupby("molecule_id", sort=False):
            cand_feats = self.build_molecule_candidate_features(grp)
            if cand_feats.empty:
                ranked_smiles = fallback_smiles[:max_k]
            else:
                scores = self.score_candidates(cand_feats)
                order = np.argsort(-scores)
                ordered_df = cand_feats.iloc[order]
                ranked_smiles = deduplicate_ranked_smiles(
                    ordered_df["smiles"].tolist(),
                    ordered_df["inchikey14"].tolist(),
                    max_k=max_k,
                )
                if not ranked_smiles:
                    ranked_smiles = fallback_smiles[:max_k]

            rows.append(
                {
                    "molecule_id": mol_id,
                    "smiles": ";".join(ranked_smiles[:max_k]),
                }
            )

        return pd.DataFrame(rows)

    def save_checkpoint(self, export_dir: str | Path) -> Path:
        """
        Saves the complete trained pipeline (CandidatePool, SpectralLibraryIndex,
        FPNet, and GBDT Reranker) to `export_dir` so it can be uploaded directly
        as a public/private Kaggle Dataset for the offline submission notebook.
        """
        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.candidate_pool.save_parquet(out / "candidate_pool.parquet")
        with open(out / "casmi_pipeline_models.pkl", "wb") as f:
            pickle.dump(
                {
                    "spectral_index": self.spectral_index,
                    "fp_model": self.fp_model,
                    "reranker": self.reranker,
                    "ppm_tolerance": self.ppm_tolerance,
                    "max_candidates_per_mol": self.max_candidates_per_mol,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return out

    @classmethod
    def load_checkpoint(cls, export_dir: str | Path) -> "CASMIPipeline":
        inp = Path(export_dir)
        pool = CandidatePool.load_parquet(inp / "candidate_pool.parquet")
        with open(inp / "casmi_pipeline_models.pkl", "rb") as f:
            state = pickle.load(f)
        pipe = cls(
            candidate_pool=pool,
            spectral_index=state["spectral_index"],
            fp_model=state["fp_model"],
            ppm_tolerance=state.get("ppm_tolerance", 15.0),
            max_candidates_per_mol=state.get("max_candidates_per_mol", 300),
        )
        pipe.reranker = state.get("reranker")
        return pipe

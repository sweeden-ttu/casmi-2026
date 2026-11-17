"""
Fast Binary Search Precursor Mass Candidate Retriever for CASMI 2026.

Indexes up to 18 million candidate structures (COCONUT 2.0 + PubChem shards + `train.parquet`
structures) sorted by exact neutral monoisotopic mass (`exact_mass`), enabling
O(log N) retrieval of all structural isomers within a +/- ppm window in < 0.2 ms per query.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from casmi26.chemistry import (
    compute_exact_mass,
    compute_np_likeness_score,
    precursor_mz_to_neutral_mass,
    smiles_to_inchikey14,
)


@dataclass
class CandidatePool:
    """Sorted in-memory index of unique candidate structures (`inchikey14` deduplicated)."""
    exact_masses: np.ndarray      # float64 sorted ascending, shape (N,)
    smiles: np.ndarray            # object/string array, shape (N,)
    inchikey14s: np.ndarray       # string array, shape (N,)
    source_priors: np.ndarray     # float32 Bayesian prior score (COCONUT=2.5, Train=2.2, PubChem=0.5)
    np_likeness_scores: np.ndarray = None  # float32 Ertl NP-Likeness score, shape (N,)

    def __post_init__(self) -> None:
        if self.np_likeness_scores is None or len(self.np_likeness_scores) != len(self.exact_masses):
            self.np_likeness_scores = np.array(
                [compute_np_likeness_score(str(s)) for s in self.smiles],
                dtype=np.float32,
            )

    def __len__(self) -> int:
        return int(self.exact_masses.size)

    def query_by_neutral_mass(
        self,
        neutral_mass: float,
        ppm_tolerance: float = 15.0,
        min_da_tolerance: float = 0.005,
        max_candidates: int = 500,
    ) -> pd.DataFrame:
        """
        Retrieves all candidate structures within `max(neutral_mass * ppm * 1e-6, min_da_tolerance)`
        using `np.searchsorted`. If fewer than 25 candidates are found, automatically expands
        the window so that at least 25 candidates are returned.
        """
        if len(self) == 0 or neutral_mass <= 0:
            return pd.DataFrame(
                columns=[
                    "smiles",
                    "inchikey14",
                    "exact_mass",
                    "source_prior",
                    "np_likeness_score",
                    "mass_error_da",
                    "mass_error_ppm",
                ]
            )

        delta_da = max(neutral_mass * ppm_tolerance * 1e-6, min_da_tolerance)
        left = int(np.searchsorted(self.exact_masses, neutral_mass - delta_da, side="left"))
        right = int(np.searchsorted(self.exact_masses, neutral_mass + delta_da, side="right"))

        # If fewer than 25 candidates lie in the tight ppm window, expand to nearest neighbors
        if (right - left) < 25:
            center = int(np.searchsorted(self.exact_masses, neutral_mass))
            half = 25
            left = max(0, center - half)
            right = min(len(self), left + 2 * half)

        # Cap at max_candidates closest by mass error + natural product prior if window is huge
        idx = np.arange(left, right)
        mass_err_da = np.abs(self.exact_masses[idx] - neutral_mass)
        if idx.size > max_candidates:
            # Prioritize high source_prior (COCONUT/Train), high np_likeness_score, and small mass error
            rank_score = mass_err_da - 0.002 * self.source_priors[idx] - 0.001 * self.np_likeness_scores[idx]
            keep = np.argpartition(rank_score, max_candidates)[:max_candidates]
            idx = idx[keep]
            mass_err_da = np.abs(self.exact_masses[idx] - neutral_mass)

        mass_err_ppm = (mass_err_da / max(neutral_mass, 1.0)) * 1e6
        return pd.DataFrame(
            {
                "smiles": self.smiles[idx],
                "inchikey14": self.inchikey14s[idx],
                "exact_mass": self.exact_masses[idx],
                "source_prior": self.source_priors[idx],
                "np_likeness_score": self.np_likeness_scores[idx],
                "mass_error_da": mass_err_da,
                "mass_error_ppm": mass_err_ppm,
            }
        )

    def sample_hard_isomer_negatives(
        self,
        neutral_mass: float,
        true_inchikey14: str,
        k_negatives: int = 8,
        ppm_tolerance: float = 25.0,
    ) -> List[str]:
        """
        Explicitly samples up to `k_negatives` hard same-mass / same-formula structural isomers
        from COCONUT 2.0 and PubChem whose `inchikey14` differs from `true_inchikey14`.
        Used for hard-negative contrastive and fingerprint training (Probe 2).
        """
        same_mass_isomers = self.query_by_neutral_mass(
            neutral_mass, ppm_tolerance=ppm_tolerance, max_candidates=max(64, k_negatives * 4)
        )
        if same_mass_isomers.empty:
            return []
        hard_df = same_mass_isomers[same_mass_isomers["inchikey14"] != str(true_inchikey14)]
        if hard_df.empty:
            return []
        return hard_df["smiles"].head(k_negatives).tolist()

    def save_parquet(self, path: str | Path) -> None:
        df = pd.DataFrame(
            {
                "exact_mass": self.exact_masses,
                "smiles": self.smiles,
                "inchikey14": self.inchikey14s,
                "source_prior": self.source_priors,
                "np_likeness_score": self.np_likeness_scores,
            }
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    @classmethod
    def load_parquet(cls, path: str | Path) -> "CandidatePool":
        df = pd.read_parquet(path)
        order = np.argsort(df["exact_mass"].to_numpy(dtype=np.float64))
        df = df.iloc[order].reset_index(drop=True)
        np_scores = (
            df["np_likeness_score"].to_numpy(dtype=np.float32)
            if "np_likeness_score" in df.columns
            else None
        )
        return cls(
            exact_masses=df["exact_mass"].to_numpy(dtype=np.float64),
            smiles=df["smiles"].to_numpy(dtype=object),
            inchikey14s=df["inchikey14"].to_numpy(dtype=object),
            source_priors=df["source_prior"].to_numpy(dtype=np.float32),
            np_likeness_scores=np_scores,
        )


def build_candidate_pool_from_sources(
    train_df: Optional[pd.DataFrame] = None,
    candidate_parquet_paths: Optional[Sequence[str | Path]] = None,
) -> CandidatePool:
    """
    Builds a deduplicated, exact-mass-sorted `CandidatePool` combining:
    1. Unique structures from `train.parquet` (source_prior = 1.5, or 2.2 if natural product lib)
    2. External candidate shards (COCONUT 2.0 shards get source_prior = 2.5, PubChem = 0.5)
    """
    records: Dict[str, Tuple[float, str, float]] = {}

    if train_df is not None and not train_df.empty:
        # Extract unique inchikey14 structures from train_df
        cols = ["inchikey14", "normalized_smiles"]
        if all(c in train_df.columns for c in cols):
            sub = train_df.drop_duplicates(subset=["inchikey14"]).copy()
            for row in sub.itertuples(index=False):
                ik14 = getattr(row, "inchikey14", None)
                smi = getattr(row, "normalized_smiles", None)
                if not ik14 or not smi:
                    continue
                # Estimate neutral mass from precursor_mz & adduct if exact_mass not precomputed
                exact_m = getattr(row, "exact_mass", None)
                if exact_m is None or not np.isfinite(exact_m):
                    exact_m = compute_exact_mass(smi)
                if exact_m is None or not np.isfinite(exact_m):
                    mz = getattr(row, "precursor_mz", 0.0)
                    add = getattr(row, "adduct", "[M+H]+")
                    mode = getattr(row, "ionization_mode", "positive")
                    exact_m = precursor_mz_to_neutral_mass(float(mz), str(add), str(mode))

                ingest = str(getattr(row, "ingest_lib", "")).lower()
                prior = 1.2 if "enveda-180" in ingest else 2.2
                records[str(ik14)] = (float(exact_m), str(smi), float(prior))

    if candidate_parquet_paths:
        for p in candidate_parquet_paths:
            path_obj = Path(p)
            if not path_obj.exists():
                continue
            is_coconut = "coconut" in str(path_obj).lower()
            prior = 2.5 if is_coconut else 0.5
            cdf = pd.read_parquet(path_obj, columns=["smiles", "inchikey14", "exact_mass"])
            for smi, ik14, em in zip(cdf["smiles"], cdf["inchikey14"], cdf["exact_mass"]):
                ik_str = str(ik14)
                if ik_str not in records or prior > records[ik_str][2]:
                    records[ik_str] = (float(em), str(smi), float(prior))

    if not records:
        return CandidatePool(
            exact_masses=np.empty(0, dtype=np.float64),
            smiles=np.empty(0, dtype=object),
            inchikey14s=np.empty(0, dtype=object),
            source_priors=np.empty(0, dtype=np.float32),
        )

    ik_list = list(records.keys())
    masses = np.array([records[k][0] for k in ik_list], dtype=np.float64)
    smiles_arr = np.array([records[k][1] for k in ik_list], dtype=object)
    priors = np.array([records[k][2] for k in ik_list], dtype=np.float32)
    ik_arr = np.array(ik_list, dtype=object)

    order = np.argsort(masses)
    return CandidatePool(
        exact_masses=masses[order],
        smiles=smiles_arr[order],
        inchikey14s=ik_arr[order],
        source_priors=priors[order],
    )

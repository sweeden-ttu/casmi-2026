#!/usr/bin/env python3
"""
Automated Rubric Evaluator for Enveda CASMI 2026 Colab & Kaggle Notebooks (.ipynb / .py).

Evaluates any CASMI 2026 retrieval-and-reranking notebook or codebase on a 0-100 scale,
awarding up to 25 points for each of the four domain-critical probes:

1. Probe 1 (0-25 pts): Multi-Spectrum Consensus (`molecule_id` grouping)
   - Simultaneous exploitation of all 1-16 `spectrum_id`s per `molecule_id`
   - Exact neutral mass triangulation across adducts (`[M+H]+`, `[M+Na]+`, `[M-H]-`, etc.)
     with outlier-resistant consensus & uncertainty estimation
   - Multi-spectrum fingerprint / embedding / rank fusion (RRF / probability pooling) across collision energies

2. Probe 2 (0-25 pts): Hard-Negative Isomer Training
   - Explicit sampling of same-formula / same-exact-mass structural isomers from COCONUT / PubChem
   - Contrastive / listwise / margin loss against hard isomer negatives (rather than random batch negatives)
   - Asymmetric / focal bit weighting for sparse molecular fingerprints & leak-free `inchikey14` CV splits

3. Probe 3 (0-25 pts): Natural Product Prior Calibration
   - Explicit database membership priors (COCONUT 2.0, GNPS/MassBank natural products vs. synthetic)
   - Computation of Ertl Natural Product Likeness (`NP-Likeness`) or biosynthetic structural descriptors
     (sp3 fraction, oxygen/nitrogen ratios, aliphatic/aromatic ring topology, glycoside/terpene motifs)
   - Integration of natural product priors into candidate retrieval pruning and GBDT/Ranker tie-breaking

4. Probe 4 (0-25 pts): Strict `RDKit 2026.03.3` Tautomer & `InChIKey14` Deduplication
   - Explicit pinning/verification of `rdkit==2026.03.3` and `TautomerEnumerator.Canonicalize`
   - Stereochemistry stripping (`RemoveStereochemistry` / first-block 14-char `inchikey14`)
   - End-to-end 3-stage deduplication: (a) candidate pool indexing, (b) training label/fold grouping,
     and (c) final Top-25 `submission.csv` output filtering + positive-mode 0.000548 Da electron-mass fix.

Usage:
    python3 scripts/score_notebook_rubric.py notebooks/01_colab_data_and_gpu_training.ipynb --include-package
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class CriterionCheck:
    name: str
    max_points: int
    patterns: List[str]
    min_matches: int = 1
    description: str = ""


@dataclass
class ProbeResult:
    probe_id: int
    title: str
    earned_points: int
    max_points: int
    breakdown: List[Tuple[str, int, int, bool, List[str]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rubric Definition: 4 Probes x 25 Points = 100 Points Total
# ---------------------------------------------------------------------------

PROBE_1_CRITERIA: List[CriterionCheck] = [
    CriterionCheck(
        name="1A. Molecule-Level Grouping (`groupby('molecule_id')`)",
        max_points=5,
        patterns=[
            r"groupby\(\s*['\"]molecule_id['\"]",
            r"partition_by\(\s*['\"]molecule_id['\"]",
            r"mol_spectra_df",
        ],
        min_matches=1,
        description="Groups all 1-16 `spectrum_id` acquisitions by `molecule_id` prior to candidate scoring.",
    ),
    CriterionCheck(
        name="1B. Multi-Adduct Exact Neutral Mass Triangulation",
        max_points=7,
        patterns=[
            r"triangulate_molecule_neutral_mass",
            r"precursor_mz_to_neutral_mass",
            r"\[M\+Na\]\+",
            r"\[M\+NH4\]\+",
            r"\[M-H\]-",
            r"ADDUCT_MASS_SHIFTS",
        ],
        min_matches=3,
        description="Converts diverse adducts ([M+H]+, [M+Na]+, [M-H]-, etc.) to neutral mass and computes outlier-resistant consensus.",
    ),
    CriterionCheck(
        name="1C. Multi-Spectrum Fingerprint / Embedding Consensus Across Collision Energies",
        max_points=7,
        patterns=[
            r"consensus_pred_fp",
            r"pred_fps_per_spec",
            r"collision_energy",
            r"np\.mean\(\s*pred_fps",
            r"multi_spectrum_fp_pool",
        ],
        min_matches=2,
        description="Predicts fingerprints or embeddings per spectrum_id and pools/weights them across collision energies.",
    ),
    CriterionCheck(
        name="1D. Multi-Spectrum Spectral Similarity & Reciprocal Rank / Score Fusion",
        max_points=6,
        patterns=[
            r"lib_max_entropy_sim",
            r"lib_mean_entropy_sim",
            r"reciprocal_rank_fusion|rrf_score",
            r"num_spectra_for_mol",
            r"neutral_mass_std_da",
        ],
        min_matches=3,
        description="Aggregates max/mean spectral entropy similarity, RRF score, and mass-triangulation uncertainty across spectra.",
    ),
]

PROBE_2_CRITERIA: List[CriterionCheck] = [
    CriterionCheck(
        name="2A. Same-Mass / Same-Formula Hard Isomer Negative Sampling",
        max_points=8,
        patterns=[
            r"sample_hard_isomer_negatives",
            r"hard_neg_fps|hard_negatives",
            r"query_by_neutral_mass",
            r"same_mass_isomers",
        ],
        min_matches=2,
        description="Explicitly mines same-mass structural isomers from COCONUT/PubChem as hard negatives for each training molecule.",
    ),
    CriterionCheck(
        name="2B. Cross-Modal Contrastive / Margin Ranking Loss Against Hard Isomers",
        max_points=7,
        patterns=[
            r"HardIsomerContrastiveLoss|hard_isomer_contrastive_loss",
            r"InfoNCE|MarginRankingLoss|TripletMarginLoss|cosine_embedding_loss",
            r"hard_isomer_margin",
            r"lambdarank",
        ],
        min_matches=2,
        description="Trains neural spectrum-to-structure / reranking models with contrastive InfoNCE, margin, or LambdaRank loss over hard isomers.",
    ),
    CriterionCheck(
        name="2C. Asymmetric / Pos-Weighted Fingerprint Loss for Sparse Bit Prediction",
        max_points=5,
        patterns=[
            r"AsymmetricFingerprintLoss|asymmetric_focal_bce_loss",
            r"BCEWithLogitsLoss\(\s*pos_weight",
            r"calibrated_fp_log_likelihood",
        ],
        min_matches=2,
        description="Addresses sparse fingerprint bit imbalance during training and uses Platt/log-likelihood bit scoring at inference.",
    ),
    CriterionCheck(
        name="2D. Leak-Free `inchikey14` CV Split & Stratified Class-2 Isomer Validation",
        max_points=5,
        patterns=[
            r"assign_leak_free_folds",
            r"group_col\s*=\s*['\"]inchikey14['\"]",
            r"sample_timstof_like_validation_panel",
            r"Class\s*2|out-of-library",
        ],
        min_matches=2,
        description="Prevents stereoisomer/tautomer leakage across folds and evaluates out-of-library isomer ranking (Class 2).",
    ),
]

PROBE_3_CRITERIA: List[CriterionCheck] = [
    CriterionCheck(
        name="3A. COCONUT 2.0 & Natural Product Library Membership Priors",
        max_points=7,
        patterns=[
            r"coconut",
            r"source_prior",
            r"is_coconut",
            r"enveda-180",
        ],
        min_matches=3,
        description="Assigns explicit higher Bayesian source priors to COCONUT 2.0 and natural product libraries over synthetic shards.",
    ),
    CriterionCheck(
        name="3B. Ertl Natural Product Likeness (`NP-Likeness`) & Biosynthetic Descriptors",
        max_points=8,
        patterns=[
            r"compute_np_likeness_score|np_likeness",
            r"rdMolDescriptors\.CalcNum",
            r"CalcFractionCSP3|fraction_csp3",
            r"oxygen_to_carbon|heteroatom_ratio|biosynthetic",
        ],
        min_matches=2,
        description="Computes Ertl NP-Likeness score and biosynthetic structural features (FractionCSP3, O/C ratio, ring systems, chiral density).",
    ),
    CriterionCheck(
        name="3C. Prior-Guided Candidate Pool Pruning & Tie-Breaking in Reranker",
        max_points=5,
        patterns=[
            r"np_likeness_score",
            r"source_prior",
            r"rank_score\s*=.*source_prior",
        ],
        min_matches=2,
        description="Uses COCONUT membership and NP-Likeness both during candidate window capping and inside the final GBDT/linear reranker.",
    ),
    CriterionCheck(
        name="3D. Analog Spectral Network Propagation (`Network Rerank`)",
        max_points=5,
        patterns=[
            r"network_max_tanimoto",
            r"network_weighted_tanimoto",
            r"top_analogs",
            r"analog_mass_window_da",
        ],
        min_matches=2,
        description="Propagates structural similarity from top spectral neighbors in train.parquet to break ties among novel natural product analogs.",
    ),
]

PROBE_4_CRITERIA: List[CriterionCheck] = [
    CriterionCheck(
        name="4A. Pinned `RDKit 2026.03.3` & `TautomerEnumerator.Canonicalize`",
        max_points=7,
        patterns=[
            r"2026\.03\.3",
            r"TautomerEnumerator|rdTautomerer",
            r"Canonicalize\(",
            r"canonicalize_tautomer_smiles",
        ],
        min_matches=3,
        description="Pins RDKit 2026.03.3 and applies RDKit tautomer canonicalization before computing InChIKeys or comparing structures.",
    ),
    CriterionCheck(
        name="4B. Stereochemistry Stripping & First-Block `InChIKey14` Extraction",
        max_points=6,
        patterns=[
            r"RemoveStereochemistry",
            r"isomericSmiles\s*=\s*False",
            r"MolToInChIKey",
            r"\[:14\]|split\(['\"]-['\"]\)\[0\]",
            r"smiles_to_inchikey14",
        ],
        min_matches=3,
        description="Strips stereocenters (R/S, E/Z) and extracts the 14-character planar connectivity InChIKey block (`inchikey14`).",
    ),
    CriterionCheck(
        name="4C. 3-Stage `InChIKey14` Deduplication (Index, Training Labels, Top-25 Submission)",
        max_points=7,
        patterns=[
            r"deduplicate_ranked_smiles",
            r"drop_duplicates\(\s*subset=\[\s*['\"]inchikey14['\"]\s*\]",
            r"seen_keys",
            r"score_mrr_at_25",
        ],
        min_matches=3,
        description="Deduplicates tautomers/stereoisomers by inchikey14 during CandidatePool indexing, training, and final Top-25 submission generation.",
    ),
    CriterionCheck(
        name="4D. Positive-Mode `0.000548 Da` Electron-Mass Calibration Fix",
        max_points=5,
        patterns=[
            r"0\.000548",
            r"ELECTRON_MASS_DA",
            r"apply_electron_mass_correction",
        ],
        min_matches=2,
        description="Accounts for the 0.00054858 Da electron mass in positive/negative adduct shifts and Enveda-180 calibration.",
    ),
]

ALL_PROBES: List[Tuple[int, str, List[CriterionCheck]]] = [
    (1, "Multi-Spectrum Consensus (molecule_id grouping)", PROBE_1_CRITERIA),
    (2, "Hard-Negative Isomer Training", PROBE_2_CRITERIA),
    (3, "Natural Product Prior Calibration", PROBE_3_CRITERIA),
    (4, "Strict RDKit 2026.03.3 Tautomer & InChIKey14 Deduplication", PROBE_4_CRITERIA),
]


# ---------------------------------------------------------------------------
# Source Extraction & Scoring Engine
# ---------------------------------------------------------------------------

def extract_code_and_markdown_from_path(path: Path) -> str:
    """Extracts source text from a `.ipynb` notebook or `.py` file."""
    if not path.exists():
        return ""
    if path.suffix.lower() == ".ipynb":
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
            chunks: List[str] = []
            for cell in nb.get("cells", []):
                src = cell.get("source", [])
                if isinstance(src, list):
                    chunks.append("".join(src))
                elif isinstance(src, str):
                    chunks.append(src)
            return "\n\n".join(chunks)
        except Exception as exc:
            return f"# Error reading {path}: {exc}"
    return path.read_text(encoding="utf-8", errors="replace")


def collect_target_corpus(
    target_paths: List[Path],
    include_imported_casmi26_package: bool = True,
) -> Tuple[str, List[Path]]:
    """
    Collects the source text of the target notebook(s) and, if the notebook imports
    from the local `casmi26` package (standard modular Colab/Kaggle design), optionally
    includes the `casmi26/*.py` module sources that the notebook executes.
    """
    resolved_files: List[Path] = []
    texts: List[str] = []

    for p in target_paths:
        if p.is_dir():
            for sub in sorted(p.rglob("*.py")) + sorted(p.rglob("*.ipynb")):
                resolved_files.append(sub)
                texts.append(extract_code_and_markdown_from_path(sub))
        elif p.exists():
            resolved_files.append(p)
            texts.append(extract_code_and_markdown_from_path(p))

    combined = "\n\n".join(texts)

    # If the notebook imports `casmi26`, include `casmi26/*.py` so modular notebooks
    # are evaluated on the full code they execute.
    if include_imported_casmi26_package and re.search(r"(from\s+casmi26|import\s+casmi26)", combined):
        repo_root = Path(__file__).resolve().parents[1]
        pkg_dir = repo_root / "casmi26"
        if pkg_dir.exists():
            for py_file in sorted(pkg_dir.glob("*.py")):
                if py_file not in resolved_files:
                    resolved_files.append(py_file)
                    texts.append(extract_code_and_markdown_from_path(py_file))
            combined = "\n\n".join(texts)

    return combined, resolved_files


def evaluate_corpus(corpus: str) -> Tuple[int, List[ProbeResult]]:
    """Scores a source corpus across all 4 probes (0-100 scale)."""
    probe_results: List[ProbeResult] = []
    total_score = 0

    for probe_id, title, criteria in ALL_PROBES:
        probe_earned = 0
        probe_max = sum(c.max_points for c in criteria)
        breakdown = []

        for crit in criteria:
            matched_patterns: List[str] = []
            for pat in crit.patterns:
                if re.search(pat, corpus, flags=re.IGNORECASE | re.MULTILINE):
                    matched_patterns.append(pat)

            n_matched = len(matched_patterns)
            if n_matched >= crit.min_matches:
                pts = crit.max_points
                passed = True
            elif n_matched > 0:
                # Partial credit proportional to matched indicators
                pts = int(round(crit.max_points * (n_matched / float(crit.min_matches))))
                passed = False
            else:
                pts = 0
                passed = False

            probe_earned += pts
            breakdown.append((crit.name, pts, crit.max_points, passed, matched_patterns))

        total_score += probe_earned
        probe_results.append(
            ProbeResult(
                probe_id=probe_id,
                title=title,
                earned_points=probe_earned,
                max_points=probe_max,
                breakdown=breakdown,
            )
        )

    return total_score, probe_results


def format_rubric_report(
    target_label: str,
    files_inspected: List[Path],
    total_score: int,
    probe_results: List[ProbeResult],
) -> str:
    """Formats a human-readable scorecard table and diagnostic report."""
    lines = [
        "=" * 84,
        f"ENVEDA CASMI 2026 — RETRIEVAL & RERANKING RUBRIC SCORECARD",
        "=" * 84,
        f"Target Evaluated : {target_label}",
        f"Files Inspected  : {', '.join(p.name for p in files_inspected)}",
        f"TOTAL SCORE      : {total_score} / 100",
        "-" * 84,
    ]

    for pr in probe_results:
        status = "PASS" if pr.earned_points == pr.max_points else ("PARTIAL" if pr.earned_points > 0 else "MISSING")
        lines.append(f"\n[Probe {pr.probe_id}] {pr.title}: {pr.earned_points} / {pr.max_points} pts ({status})")
        for crit_name, earned, max_p, passed, matched in pr.breakdown:
            mark = "[x]" if passed else ("[~]" if earned > 0 else "[ ]")
            lines.append(f"  {mark} {crit_name:<64} {earned:>2}/{max_p:<2} pts")

    lines.append("\n" + "=" * 84)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score any Colab/Kaggle .ipynb file on the 100-pt CASMI 2026 Rubric.")
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        help="Path(s) to .ipynb notebook(s) or .py files to score.",
    )
    parser.add_argument(
        "--standalone-only",
        action="store_true",
        help="Score only the raw text inside the .ipynb file without following `import casmi26` modules.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON scorecard output.",
    )
    args = parser.parse_args()

    corpus, files = collect_target_corpus(
        args.targets,
        include_imported_casmi26_package=not args.standalone_only,
    )
    total_score, probe_results = evaluate_corpus(corpus)
    report = format_rubric_report(
        target_label=", ".join(str(t) for t in args.targets),
        files_inspected=files,
        total_score=total_score,
        probe_results=probe_results,
    )
    print(report)

    if args.json_out is not None:
        payload = {
            "targets": [str(t) for t in args.targets],
            "total_score": total_score,
            "max_score": 100,
            "probes": [
                {
                    "probe_id": pr.probe_id,
                    "title": pr.title,
                    "earned_points": pr.earned_points,
                    "max_points": pr.max_points,
                    "criteria": [
                        {
                            "name": c_name,
                            "earned": earned,
                            "max": max_p,
                            "passed": passed,
                        }
                        for c_name, earned, max_p, passed, _ in pr.breakdown
                    ],
                }
                for pr in probe_results
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

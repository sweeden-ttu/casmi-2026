# [Colab Q-RL Cell 1] Dense Prefix Mass Potential & Multi-Objective Terminal Chemical Reward Engine
import math
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Approximate minimum mass contribution (atom + minimum typical implicit H) per SMILES character
TOKEN_HEAVY_MASS_LOWER_BOUND = {
    "C": 12.0000, "c": 12.0000,
    "N": 14.0031, "n": 14.0031,
    "O": 15.9949, "o": 15.9949,
    "S": 31.9721, "s": 31.9721,
    "P": 30.9738, "p": 30.9738,
    "F": 18.9984,
    "Cl": 34.9689,
    "Br": 78.9183,
    "I": 126.9045,
    # The notebook-1 tokenizer is character-level: "Cl" arrives as "C" + "l" and "Br" as "B" + "r".
    # The two-letter keys above never matched, so Cl counted as 12 Da and Br as 0. Per-character
    # increments keep the lower bound exact: C + l = Cl, B (boron) + r = Br.
    "l": 34.9689 - 12.0000,
    "B": 11.0093,
    "r": 78.9183 - 11.0093,
    "H": 1.0078,  # explicit bracket hydrogens, e.g. [nH], [NH+]
}


class CASMI26ChemicalRewardEngine:
    """
    Computes:
      1. Dense prefix potential Phi(s_t) from partial heavy-atom mass accumulation and bracket/ring balance.
      2. Terminal reward R_term(y) at [EOS] combining:
         - r_valid:   RDKit 2026.03.3 parseability + tautomer canonicalization
         - r_mass:    Gaussian + Laplace reward on exact monoisotopic mass vs. neutral precursor mass
         - r_fg71:    Cosine + L1 agreement with Mass2SMILES 71-D functional group & heteroatom predictions
         - r_formula: Bonus if Hill formula matches one of the top-3 MIST-CF formula priors
         - r_emb:     Unimodal cosine similarity cos(E^m_gen, E^m_target) in ChemFormer space
    """

    def __init__(
        self,
        tokenizer,
        canonicalize_fn,
        fg_extractor_fn,
        mass_sigma_da: float = W("nb5.reward.mass_sigma_da"),
        coarse_mass_scale_da: float = W("nb5.reward.coarse_mass_scale_da"),
    ):
        self.tokenizer = tokenizer
        self.canonicalize_fn = canonicalize_fn
        self.fg_extractor_fn = fg_extractor_fn
        self.mass_sigma_da = mass_sigma_da
        self.coarse_mass_scale_da = coarse_mass_scale_da

        # Precompute per-token heavy-atom mass table aligned with tokenizer vocabulary
        self.token_masses = np.zeros(len(tokenizer), dtype=np.float32)
        for tok_str, idx in tokenizer.stoi.items():
            if tok_str in TOKEN_HEAVY_MASS_LOWER_BOUND:
                self.token_masses[idx] = TOKEN_HEAVY_MASS_LOWER_BOUND[tok_str]

    def compute_prefix_mass_and_potential(
        self,
        token_ids: torch.Tensor,       # (B, L)
        target_neutral_mass: torch.Tensor,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes cumulative heavy-atom mass at every token step t in 1..L and returns:
          - rem_mass_ratio: (B, L) normalized remaining mass budget (m_target - m_heavy_t) / m_target
          - phi_potential:  (B, L) potential function Phi(s_t) penalizing mass budget overruns
        """
        device = token_ids.device
        tok_mass_t = torch.tensor(self.token_masses, device=device)[token_ids]  # (B, L)
        cum_heavy_mass = torch.cumsum(tok_mass_t, dim=1)                        # (B, L)

        target_m = target_neutral_mass.unsqueeze(1).clamp(min=50.0)             # (B, 1)
        # Since hydrogens are implicit in SMILES, heavy atoms typically account for ~90-94% of total mass
        expected_heavy_target = W("nb5.potential.heavy_fraction") * target_m
        rem_mass = target_m - cum_heavy_mass
        rem_mass_ratio = torch.clamp(rem_mass / target_m, min=-1.0, max=1.0)

        # Smooth potential peaks as cumulative heavy mass approaches expected_heavy_target,
        # and drops sharply if heavy-atom mass alone exceeds target_neutral_mass + 1.0 Da
        overrun_da = F.relu(cum_heavy_mass - (target_m + W("nb5.potential.overrun_margin_da")))
        underrun_frac = torch.abs(cum_heavy_mass - expected_heavy_target) / target_m
        phi_potential = torch.exp(-W("nb5.potential.underrun_decay") * underrun_frac) - W("nb5.potential.overrun_penalty") * overrun_da

        return rem_mass_ratio, phi_potential

    @torch.no_grad()
    def evaluate_terminal_smiles_batch(
        self,
        generated_seqs: List[List[int]],
        target_neutral_masses: np.ndarray,          # (B,)
        target_fg71_preds: np.ndarray,              # (B, 71)
        formula_priors_batch: Optional[List[List[str]]] = None,
        mol_encoder: Optional[nn.Module] = None,
        target_mol_embs: Optional[torch.Tensor] = None,  # (B, 256) ground-truth or Stage-1 anchor
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, List[Optional[str]], List[Optional[str]]]:
        """
        Computes scalar terminal reward R_term in [-1.0, +5.0] for each decoded sequence in batch.
        """
        B = len(generated_seqs)
        rewards = np.zeros(B, dtype=np.float32)
        canon_smiles_out: List[Optional[str]] = []
        ik14_out: List[Optional[str]] = []

        valid_indices = []
        valid_tok_Batch = []

        for b in range(B):
            raw_smi = self.tokenizer.decode(generated_seqs[b])
            canon_smi, ik14 = self.canonicalize_fn(raw_smi)
            canon_smiles_out.append(canon_smi)
            ik14_out.append(ik14)

            if canon_smi is None:
                rewards[b] = W("nb5.reward.invalid")  # Invalid SMILES syntax or rejected valence
                continue

            mol = Chem.MolFromSmiles(canon_smi)
            if mol is None:
                rewards[b] = W("nb5.reward.invalid")
                continue

            # 1. Chemical Validity Base Reward (+0.50)
            r_total = W("nb5.reward.valid_base")

            # 2. Exact Monoisotopic Precursor Mass Reward (up to +2.00)
            # Combines a coarse exponential funnel (guides within +/- 14 Da) and a fine sub-15 ppm peak
            gen_mass = float(Descriptors.ExactMolWt(mol))
            target_m = float(target_neutral_masses[b])
            mass_diff_da = abs(gen_mass - target_m)
            ppm_err = (mass_diff_da / max(target_m, 100.0)) * 1e6

            r_mass_coarse = math.exp(-mass_diff_da / self.coarse_mass_scale_da)
            r_mass_fine = math.exp(-0.5 * (mass_diff_da / self.mass_sigma_da) ** 2)
            r_mass = W("nb5.reward.mass_coarse") * r_mass_coarse + W("nb5.reward.mass_fine") * r_mass_fine
            r_total += r_mass

            # 3. MIST-CF / SIRIUS Molecular Formula Match Bonus (+0.75)
            if formula_priors_batch is not None and b < len(formula_priors_batch):
                gen_formula = rdMolDescriptors.CalcMolFormula(mol)
                priors = formula_priors_batch[b]
                if priors and gen_formula == priors[0]:
                    r_total += W("nb5.reward.formula_top1")
                elif priors and gen_formula in priors[:3]:
                    r_total += W("nb5.reward.formula_top3")

            # 4. Mass2SMILES 71-D Functional Group + Heteroatom Consistency Reward (up to +1.00)
            _, gen_fg71 = self.fg_extractor_fn(canon_smi)
            if gen_fg71 is not None:
                pred_fg71 = target_fg71_preds[b]
                # Separate functional groups (0..59) and heteroatoms/rings (60..70)
                fg_l1 = float(np.mean(np.abs(gen_fg71[:60] - pred_fg71[:60])))
                hetero_l1 = float(np.mean(np.abs(gen_fg71[60:] - pred_fg71[60:])))
                r_fg = (W("nb5.reward.fg_weight") * math.exp(-W("nb5.reward.fg_decay") * fg_l1)
                        + W("nb5.reward.hetero_weight") * math.exp(-W("nb5.reward.hetero_decay") * hetero_l1))
                r_total += r_fg

            rewards[b] = r_total
            if mol_encoder is not None and target_mol_embs is not None:
                valid_indices.append(b)
                valid_tok_Batch.append(self.tokenizer.encode(canon_smi, add_cls=True))

        # 5. Unimodal ChemFormer Embedding Cosine Similarity Bonus (up to +0.75)
        if valid_indices and mol_encoder is not None and target_mol_embs is not None:
            tok_tensor = torch.stack(valid_tok_Batch).to(device)
            _, e_m_gen = mol_encoder(tok_tensor)  # (N_val, 256)
            target_sub = target_mol_embs[valid_indices].to(device)
            cos_sims = torch.sum(e_m_gen * target_sub, dim=-1).clamp(min=0.0).cpu().numpy()
            for idx_pos, b_idx in enumerate(valid_indices):
                rewards[b_idx] += W("nb5.reward.embedding_bonus") * float(cos_sims[idx_pos])

        return torch.tensor(rewards, dtype=torch.float32, device=device), canon_smiles_out, ik14_out
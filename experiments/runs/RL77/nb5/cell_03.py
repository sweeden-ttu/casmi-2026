# [Colab Q-RL Cell 3] Stage 2.5 Soft Q-Learning + Conservative Q + KL-Anchored Policy Optimization
import copy


class GLMRStage25QLearningTrainer:
    """
    Fine-tunes the Stage 2 Spec-Mol Cross-Fusion Decoder using Dueling Soft Q-Learning
    and KL-regularized Advantage Actor-Critic updates.
    """

    def __init__(
        self,
        q_decoder: ChemFormerQDecoderWrapper,
        reward_engine: CASMI26ChemicalRewardEngine,
        gamma: float = 0.99,
        alpha_ent: float = 0.05,
        cql_weight: float = 0.10,
        kl_beta: float = 0.20,
        sup_anchor_weight: float = 0.25,
        polyak_tau: float = 0.01,
    ):
        self.q_decoder = q_decoder
        self.reward_engine = reward_engine
        self.gamma = gamma
        self.alpha_ent = alpha_ent
        self.cql_weight = cql_weight
        self.kl_beta = kl_beta
        self.sup_anchor_weight = sup_anchor_weight
        self.polyak_tau = polyak_tau

        # Freeze a copy of the Stage 2 supervised decoder as the KL reference policy pi_ref
        self.ref_decoder = copy.deepcopy(q_decoder.base_decoder).eval()
        for p in self.ref_decoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def sample_on_policy_rollouts(
        self,
        memory: torch.Tensor,              # (B, P, D) H^ca
        target_neutral_mass: torch.Tensor, # (B,)
        max_len: int = 96,
        temperature: float = 0.85,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """Samples on-policy SMILES trajectories y ~ pi_phi( . | H^ca) for RL exploration."""
        self.q_decoder.eval()
        B = memory.size(0)
        device = memory.device
        bos_id = self.q_decoder.base_decoder.bos_id
        eos_id = self.q_decoder.base_decoder.eos_id
        pad_id = self.q_decoder.base_decoder.pad_id

        seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.q_decoder.base_decoder(seqs, memory)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tok = torch.where(finished, torch.tensor(pad_id, device=device), next_tok)
            seqs = torch.cat([seqs, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_id)
            if torch.all(finished):
                break

        decoded_lists = [row.cpu().tolist() for row in seqs]
        self.q_decoder.train()
        return seqs, decoded_lists

    def compute_shaped_td_and_actor_losses(
        self,
        traj_ids: torch.Tensor,             # (B, Q) token trajectory [BOS, y_1, ..., EOS, PAD]
        memory: torch.Tensor,               # (B, P, D) H^ca
        target_neutral_mass: torch.Tensor,  # (B,)
        terminal_rewards: torch.Tensor,     # (B,) R_term(y)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes:
          1. Soft Bellman TD loss + CQL penalty on Q(s_t, a_t)
          2. Advantage-weighted Actor loss + KL(pi_phi || pi_ref) penalty
        """
        pad_id = self.q_decoder.base_decoder.pad_id
        eos_id = self.q_decoder.base_decoder.eos_id

        inp_ids = traj_ids[:, :-1]   # (B, T) states s_t
        act_ids = traj_ids[:, 1:]    # (B, T) actions a_t
        valid_mask = (act_ids != pad_id).float()  # (B, T)

        # Compute dense prefix potentials Phi(s_t) and Phi(s_{t+1})
        rem_ratio_all, phi_all = self.reward_engine.compute_prefix_mass_and_potential(
            traj_ids, target_neutral_mass
        )
        rem_s = rem_ratio_all[:, :-1]
        rem_next = rem_ratio_all[:, 1:]
        phi_s = phi_all[:, :-1]
        phi_next = phi_all[:, 1:]

        # Forward current policy & Q-network on s_t
        policy_logits, q_all, v_s = self.q_decoder.forward_policy_and_q(
            inp_ids, memory, rem_s, target_neutral_mass, use_target_q=False
        )
        q_taken = q_all.gather(dim=-1, index=act_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Forward target Q-network on s_{t+1} for Soft Bellman backup
        with torch.no_grad():
            _, q_next_all, _ = self.q_decoder.forward_policy_and_q(
                act_ids, memory, rem_next, target_neutral_mass, use_target_q=True
            )
            # Soft value V_targ(s_{t+1}) = alpha * logsumexp(Q_targ(s_{t+1}, a') / alpha)
            v_next_soft = self.alpha_ent * torch.logsumexp(
                q_next_all / self.alpha_ent, dim=-1
            )

            # Identify terminal step for each sequence (where act_ids == eos_id or last valid step)
            is_terminal = (act_ids == eos_id) | (
                torch.cat([valid_mask[:, 1:], torch.zeros_like(valid_mask[:, :1])], dim=1) == 0.0
            )
            is_term_f = is_terminal.float()

            # Potential-based reward shaping: r_t = gamma * Phi(s_{t+1}) - Phi(s_t) on non-terminal steps,
            # and r_T = R_term(y) - Phi(s_T) on the terminal step
            step_rewards = (1.0 - is_term_f) * (self.gamma * phi_next - phi_s) + is_term_f * (
                terminal_rewards.unsqueeze(1) - phi_s
            )
            td_targets = step_rewards + (1.0 - is_term_f) * self.gamma * v_next_soft

            # Reference policy logits for KL regularization
            ref_logits = self.ref_decoder(inp_ids, memory)

        # 1. Huber TD Loss + Conservative Q-Learning (CQL) penalty
        td_loss = F.smooth_l1_loss(q_taken, td_targets, reduction="none")
        cql_penalty = torch.logsumexp(q_all, dim=-1) - q_taken
        q_loss = torch.sum((td_loss + self.cql_weight * cql_penalty) * valid_mask) / valid_mask.sum().clamp(min=1.0)

        # 2. Advantage-Weighted Policy Gradient + KL(pi_phi || pi_ref)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        probs = F.softmax(policy_logits, dim=-1)

        taken_log_prob = log_probs.gather(dim=-1, index=act_ids.unsqueeze(-1)).squeeze(-1)
        advantages = (q_taken - v_s.squeeze(-1)).detach()
        # Normalize advantages across valid tokens in batch
        adv_mean = (advantages * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        adv_std = torch.sqrt(
            ((advantages - adv_mean).pow(2) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0) + 1e-6
        )
        norm_advantages = torch.clamp((advantages - adv_mean) / adv_std, min=-3.0, max=3.0)

        pg_loss = -taken_log_prob * norm_advantages
        kl_div = torch.sum(probs * (log_probs - ref_log_probs), dim=-1)
        actor_loss = torch.sum((pg_loss + self.kl_beta * kl_div) * valid_mask) / valid_mask.sum().clamp(min=1.0)

        return q_loss, actor_loss

    def train_rl_step(
        self,
        batch: Dict[str, torch.Tensor],
        glmr_model: nn.Module,
        fg_head: nn.Module,
        subformula_filter,
        generate_formula_priors_fn,
        retrieve_topk_fn,
        lib_masses: np.ndarray,
        lib_embs: torch.Tensor,
        adduct_shifts: Dict[str, float],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float]:
        """Executes one complete hybrid Off-Policy (Expert) + On-Policy (Rollout) Q-RL optimization step."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            h_s_mol, e_s_mol, _ = glmr_model.encode_molecule_spectra(batch)
            fg_71_pred, _ = fg_head(h_s_mol)
            _, target_mol_embs = glmr_model.mol_encoder(batch["mol_tokens_cls"])

            h_m_k, _ = retrieve_topk_fn(
                e_s_mol, batch["precursor_mz"], batch["primary_adduct"], lib_masses, lib_embs
            )
            h_ca = glmr_model.cross_fusion(h_s_mol, h_m_k)

            # Infer target neutral mass per molecule in batch
            B = h_ca.size(0)
            neutral_masses = []
            for b in range(B):
                pmz = float(batch["precursor_mz"][b].item())
                shift = adduct_shifts.get(str(batch["primary_adduct"][b]), -1.007276)
                neutral_masses.append(pmz + shift)
            target_neutral_mass_t = torch.tensor(neutral_masses, dtype=torch.float32, device=device)

            formula_priors = generate_formula_priors_fn(
                batch, subformula_filter, fg_71_pred_batch=fg_71_pred
            )

        # A. On-policy rollout trajectories y_hat ~ pi_phi(. | H^ca)
        rollout_seqs, rollout_lists = self.sample_on_policy_rollouts(
            h_ca, target_neutral_mass_t, max_len=96
        )
        rollout_rewards, _, _ = self.reward_engine.evaluate_terminal_smiles_batch(
            generated_seqs=rollout_lists,
            target_neutral_masses=np.asarray(neutral_masses, dtype=np.float32),
            target_fg71_preds=fg_71_pred.cpu().numpy(),
            formula_priors_batch=formula_priors,
            mol_encoder=glmr_model.mol_encoder,
            target_mol_embs=target_mol_embs,
            device=device,
        )

        q_loss_on, actor_loss_on = self.compute_shaped_td_and_actor_losses(
            rollout_seqs, h_ca, target_neutral_mass_t, rollout_rewards
        )

        # B. Expert ground-truth trajectory anchor (replay buffer / imitation anchor)
        expert_seqs = batch["mol_tokens_dec"]
        expert_rewards = torch.full((B,), 4.75, dtype=torch.float32, device=device)
        q_loss_exp, _ = self.compute_shaped_td_and_actor_losses(
            expert_seqs, h_ca, target_neutral_mass_t, expert_rewards
        )

        # Supervised token anchor loss on ground-truth SMILES to preserve syntactic grammar
        sup_logits = self.q_decoder.base_decoder(expert_seqs[:, :-1], h_ca)
        sup_loss = F.cross_entropy(
            sup_logits.reshape(-1, sup_logits.size(-1)),
            expert_seqs[:, 1:].reshape(-1),
            ignore_index=self.q_decoder.base_decoder.pad_id,
        )

        total_loss = (
            0.5 * (q_loss_on + q_loss_exp)
            + actor_loss_on
            + self.sup_anchor_weight * sup_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_decoder.parameters(), 1.0)
        optimizer.step()

        self.q_decoder.sync_target_q_heads(tau=self.polyak_tau)

        return {
            "total_loss": float(total_loss.item()),
            "q_loss": float(0.5 * (q_loss_on + q_loss_exp).item()),
            "actor_loss": float(actor_loss_on.item()),
            "mean_rollout_reward": float(rollout_rewards.mean().item()),
        }
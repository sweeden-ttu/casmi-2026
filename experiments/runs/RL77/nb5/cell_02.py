# [Colab Q-RL Cell 2] Mass-Conditioned Dueling Q-Network Wrapper for ChemFormerMolecularDecoder
class ChemFormerQDecoderWrapper(nn.Module):
    """
    Wraps ChemFormerMolecularDecoder with:
      1. Remaining-mass sinusoidal state conditioning (informs policy & Q-head how many Daltons remain).
      2. Dueling Action-Value Head Q(s_t, a_t) = V(s_t) + (A(s_t, a_t) - mean_a' A(s_t, a')).
      3. EMA Target Q-Network parameters for stable Bellman temporal-difference backups.
    """

    def __init__(
        self,
        base_decoder: nn.Module,
        d_model: int = 256,
        mass_emb_dim: int = 32,
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.vocab_size = base_decoder.out_proj.out_features
        self.d_model = d_model

        # Project [remaining_mass_ratio, cumulative_mass / 1000.0] into mass_emb_dim
        self.mass_budget_proj = nn.Sequential(
            nn.Linear(2, mass_emb_dim),
            nn.LayerNorm(mass_emb_dim),
            nn.GELU(),
        )

        # Dueling Q-Network: State Value V(s_t) and Token Advantage A(s_t, a_t)
        fused_dim = d_model + mass_emb_dim
        self.value_head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, self.vocab_size),
        )

        # Target network for stable TD backups (updated via Polyak averaging)
        self.target_value_head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.target_advantage_head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, self.vocab_size),
        )
        self.sync_target_q_heads(tau=1.0)

    @torch.no_grad()
    def sync_target_q_heads(self, tau: float = 0.005) -> None:
        """Polyak soft update of target Q-heads: theta_targ <- tau * theta + (1 - tau) * theta_targ."""
        for p, p_targ in zip(self.value_head.parameters(), self.target_value_head.parameters()):
            p_targ.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        for p, p_targ in zip(self.advantage_head.parameters(), self.target_advantage_head.parameters()):
            p_targ.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def extract_decoder_hidden_states(
        self, tgt_ids: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Runs the causal Transformer stack of base_decoder and returns hidden states (B, Q, D)."""
        B, Q = tgt_ids.shape
        positions = torch.arange(Q, device=tgt_ids.device).unsqueeze(0).expand(B, Q)
        tgt = self.base_decoder.token_emb(tgt_ids) + self.base_decoder.pos_emb(positions)
        causal_mask = torch.triu(torch.ones(Q, Q, dtype=torch.bool, device=tgt_ids.device), diagonal=1)
        tgt_pad_mask = tgt_ids == self.base_decoder.pad_id
        h_dec = self.base_decoder.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return h_dec

    def forward_policy_and_q(
        self,
        tgt_ids: torch.Tensor,            # (B, Q)
        memory: torch.Tensor,             # (B, P, D) cross-fused H^ca
        rem_mass_ratio: torch.Tensor,     # (B, Q)
        target_neutral_mass: torch.Tensor,  # (B,)
        use_target_q: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          - policy_logits: (B, Q, V) from base_decoder.out_proj
          - q_values:      (B, Q, V) action-values Q(s_t, a_t) for every token in vocabulary
          - state_values:  (B, Q, 1) V(s_t)
        """
        h_dec = self.extract_decoder_hidden_states(tgt_ids, memory)
        policy_logits = self.base_decoder.out_proj(h_dec)

        norm_target_m = (target_neutral_mass / 1000.0).unsqueeze(1).expand_as(rem_mass_ratio)
        mass_feats = torch.stack([rem_mass_ratio, norm_target_m], dim=-1)  # (B, Q, 2)
        mass_emb = self.mass_budget_proj(mass_feats)                        # (B, Q, mass_emb_dim)

        fused_state = torch.cat([h_dec, mass_emb], dim=-1)                  # (B, Q, D + mass_emb_dim)

        if use_target_q:
            v_s = self.target_value_head(fused_state)
            adv = self.target_advantage_head(fused_state)
        else:
            v_s = self.value_head(fused_state)
            adv = self.advantage_head(fused_state)

        q_values = v_s + (adv - adv.mean(dim=-1, keepdim=True))
        return policy_logits, q_values, v_s
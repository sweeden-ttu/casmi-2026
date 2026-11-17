# [Colab Cell 5] Complete PyTorch GLMR Architecture + Multi-Spectrum Aggregator
class MultiScaleSinusoidalMzEmbedding(nn.Module):
    """
    256-dimensional multi-scale sinusoidal encoding for high-resolution m/z & neutral losses
    (2-decimal precision, Voronov et al. / Mass2SMILES).
    """

    def __init__(self, d_model: int = CFG.D_MODEL, min_freq: float = 1e-2, max_freq: float = 1e3):
        super().__init__()
        assert d_model % 2 == 0
        freqs = torch.logspace(
            math.log10(min_freq), math.log10(max_freq), d_model // 2
        )
        self.register_buffer("freqs", freqs * 2.0 * math.pi)

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        # mz: (..., P) -> returns (..., P, d_model)
        angles = mz.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class SinusoidalSpectralEncoder(nn.Module):
    """
    GLMR Spectral Encoder f_s(Spec; eta):
      - Projects 256-d sinusoidal m/z + 256-d sinusoidal neutral loss + intensity + metadata
      - Passes through 6 Multi-Head Attention encoder layers (d_model = 256)
      - Outputs sequence representations H^s and mean-pooled spectrum embedding E^s
    """

    def __init__(
        self,
        d_model: int = CFG.D_MODEL,
        n_heads: int = CFG.N_HEADS,
        num_layers: int = CFG.SPEC_ENC_LAYERS,
    ):
        super().__init__()
        self.sin_emb = MultiScaleSinusoidalMzEmbedding(d_model)
        self.adduct_emb = nn.Embedding(len(ADDUCT_VOCAB), 32)
        self.ce_proj = nn.Linear(2, 32)  # [collision_energy_ev / 100.0, log10(BPI) / 10.0]

        # Input per peak: 256 (m/z) + 256 (neutral loss) + 1 (intensity) + 64 (metadata) = 577
        self.input_proj = nn.Sequential(
            nn.Linear(d_model * 2 + 1 + 64, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        mzs: torch.Tensor,
        ints: torch.Tensor,
        nls: torch.Tensor,
        ce: torch.Tensor,
        adduct: torch.Tensor,
        bpi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inputs: (B_spec, P)
        mz_feat = self.sin_emb(mzs)
        nl_feat = self.sin_emb(nls)
        meta = torch.cat(
            [
                self.adduct_emb(adduct),
                self.ce_proj(torch.stack([ce / 100.0, bpi / 10.0], dim=-1)),
            ],
            dim=-1,
        )
        meta_expanded = meta.unsqueeze(1).expand(-1, mzs.size(1), -1)

        x = torch.cat([mz_feat, nl_feat, ints.unsqueeze(-1), meta_expanded], dim=-1)
        h_s = self.input_proj(x)

        peak_padding_mask = mzs <= 0.0
        peak_padding_mask[:, 0] = False  # Precursor token at index 0 is always valid
        h_s = self.transformer(h_s, src_key_padding_mask=peak_padding_mask)

        valid_counts = (~peak_padding_mask).float().sum(dim=1, keepdim=True).clamp(min=1.0)
        masked_h_s = h_s.masked_fill(peak_padding_mask.unsqueeze(-1), 0.0)
        e_s = masked_h_s.sum(dim=1) / valid_counts
        return h_s, F.normalize(e_s, p=2, dim=-1)


class MultiSpectrumMoleculeAggregator(nn.Module):
    """
    Aggregates variable numbers of spectra (1..S) per molecule_id across collision energies
    and adducts using learned attention weights + signal-quality gating.
    """

    def __init__(self, d_model: int = CFG.D_MODEL, n_heads: int = CFG.N_HEADS):
        super().__init__()
        self.spec_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.score_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.norm = nn.LayerNorm(d_model)
        self.fg_head = nn.Linear(d_model, CFG.NUM_FG_CLASSES)  # Mass2SMILES auxiliary head

    def forward(
        self,
        h_s_all: torch.Tensor,   # (B, S, P, D)
        e_s_all: torch.Tensor,   # (B, S, D)
        spec_mask: torch.Tensor, # (B, S) True where spectrum is valid
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, P, D = h_s_all.shape
        key_padding_mask = ~spec_mask

        # Cross-spectrum self-attention over spectrum-level embeddings
        attn_out, _ = self.spec_attn(
            e_s_all, e_s_all, e_s_all, key_padding_mask=key_padding_mask
        )
        e_s_refined = self.norm(e_s_all + attn_out)

        # Compute attention weights alpha_s over the S spectra of each molecule
        logits = self.score_gate(e_s_refined).squeeze(-1)  # (B, S)
        logits = logits.masked_fill(key_padding_mask, -1e9)
        weights = F.softmax(logits, dim=-1)                # (B, S)

        # Unified molecule-level spectral embedding E^s_mol: (B, D)
        e_mol = torch.sum(e_s_refined * weights.unsqueeze(-1), dim=1)
        e_mol = F.normalize(e_mol, p=2, dim=-1)

        # Weighted pooling of peak token states H^s_mol: (B, P, D) for Stage 2 Cross-Fusion
        h_mol = torch.sum(h_s_all * weights.view(B, S, 1, 1), dim=1)
        fg_pred = self.fg_head(e_mol)
        return h_mol, e_mol, fg_pred


class ChemFormerMolecularEncoder(nn.Module):
    """
    GLMR Molecular Encoder f_m(Mol; gamma):
      - 4 bidirectional encoder layers (d_model = 256)
      - [CLS] token at index 0 produces global molecular embedding E^m = H^m_[CLS]
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = CFG.D_MODEL,
        n_heads: int = CFG.N_HEADS,
        num_layers: int = CFG.MOL_ENC_LAYERS,
        max_len: int = CFG.MAX_SMILES_LEN,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        pad_mask = input_ids == self.pad_id
        h_m = self.encoder(x, src_key_padding_mask=pad_mask)
        e_m = F.normalize(h_m[:, 0, :], p=2, dim=-1)  # [CLS] embedding
        return h_m, e_m


class SpecMolCrossFusion(nn.Module):
    """
    GLMR Equation (7):
    H^ca = Attn(Query(H^s), Key(H^m_K)) * Value(H^m_K)
    Integrates the multi-spectrum representation H^s (Query) with the top-K
    pre-retrieved molecular candidate embeddings H^m_K (Key, Value).
    """

    def __init__(self, d_model: int = CFG.D_MODEL, n_heads: int = CFG.N_HEADS):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h_s: torch.Tensor, h_m_k: torch.Tensor) -> torch.Tensor:
        # h_s: (B, P, D), h_m_k: (B, K, D)
        attn_out, _ = self.cross_attn(query=h_s, key=h_m_k, value=h_m_k)
        x = self.norm1(h_s + attn_out)
        h_ca = self.norm2(x + self.ffn(x))
        return h_ca


class ChemFormerMolecularDecoder(nn.Module):
    """
    GLMR Molecular Decoder f_g(H^ca; phi):
      - 4 autoregressive Transformer decoder layers conditioned on H^ca
      - Supports teacher-forced training and Beam Search (beam_size=5) inference
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        d_model: int = CFG.D_MODEL,
        n_heads: int = CFG.N_HEADS,
        num_layers: int = CFG.MOL_DEC_LAYERS,
        max_len: int = CFG.MAX_SMILES_LEN,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, Q = tgt_ids.shape
        positions = torch.arange(Q, device=tgt_ids.device).unsqueeze(0).expand(B, Q)
        tgt = self.token_emb(tgt_ids) + self.pos_emb(positions)
        # Boolean causal mask: matches the boolean key-padding mask (mixing float and bool masks is deprecated).
        causal_mask = torch.triu(torch.ones(Q, Q, dtype=torch.bool, device=tgt_ids.device), diagonal=1)
        tgt_pad_mask = tgt_ids == self.pad_id
        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return self.out_proj(dec_out)

    @torch.no_grad()
    def beam_search_decode(
        self, memory: torch.Tensor, beam_size: int = CFG.BEAM_SIZE, max_len: int = 128
    ) -> List[List[int]]:
        """Generates refined SMILES sequences via beam search for each item in batch.

        All live beams of a molecule share one length, so they are decoded in a single
        batched forward pass per step instead of one pass per beam.
        """
        B = memory.size(0)
        max_len = min(max_len, self.max_len)
        best_sequences = []
        for b in range(B):
            mem_b = memory[b : b + 1]  # (1, P, D)
            beams = [([self.bos_id], 0.0)]
            completed = []

            for _ in range(max_len - 1):
                live = []
                for seq, score in beams:
                    (completed if seq[-1] == self.eos_id else live).append((seq, score))
                beams = live
                if not beams or len(completed) >= beam_size:
                    break
                inp = torch.tensor([seq for seq, _ in beams], dtype=torch.long, device=memory.device)
                logits = self.forward(inp, mem_b.expand(len(beams), -1, -1))[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                topk_lp, topk_idx = torch.topk(log_probs, beam_size, dim=-1)
                candidates = []
                for i, (seq, score) in enumerate(beams):
                    for lp, tok_id in zip(topk_lp[i].tolist(), topk_idx[i].tolist()):
                        candidates.append((seq + [tok_id], score + lp))
                candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
                beams = candidates[:beam_size]

            all_finished = completed + beams
            all_finished.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
            best_sequences.append(all_finished[0][0])
        return best_sequences


class GLMRPipeline(nn.Module):
    """Unified wrapper holding Stage 1 and Stage 2 modules."""

    def __init__(self, tokenizer: SMILESTokenizer):
        super().__init__()
        self.mol_encoder = ChemFormerMolecularEncoder(
            vocab_size=len(tokenizer), pad_id=tokenizer.pad_id
        )
        self.spec_encoder = SinusoidalSpectralEncoder()
        self.aggregator = MultiSpectrumMoleculeAggregator()
        self.cross_fusion = SpecMolCrossFusion()
        self.mol_decoder = ChemFormerMolecularDecoder(
            vocab_size=len(tokenizer),
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
        )

    def encode_molecule_spectra(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, P = batch["mzs"].shape
        mzs_flat = batch["mzs"].view(B * S, P)
        ints_flat = batch["ints"].view(B * S, P)
        nls_flat = batch["nls"].view(B * S, P)
        ce_flat = batch["ce"].view(B * S)
        add_flat = batch["adduct"].view(B * S)
        bpi_flat = batch["bpi"].view(B * S)

        h_s_flat, e_s_flat = self.spec_encoder(
            mzs_flat, ints_flat, nls_flat, ce_flat, add_flat, bpi_flat
        )
        h_s_all = h_s_flat.view(B, S, P, -1)
        e_s_all = e_s_flat.view(B, S, -1)
        return self.aggregator(h_s_all, e_s_all, batch["spec_mask"])
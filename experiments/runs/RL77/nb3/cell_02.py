# [Colab Ensemble Cell 2] 2000-Bin Siamese Metric Learning + Continuous ECFP4 Predictor
class SiameseECFP4Retriever(nn.Module):
    """
    Implements the Siamese Metric Learning + Fingerprint MLP architecture (Carrera et al., 2026):
      - Spectrum Branch: 3-layer MLP (2000 -> 1024 -> 512 -> 256, ReLU) producing L2-normalized z_s.
      - Fingerprint Branch: 3-layer MLP (2048 -> 1024 -> 512 -> 256, ReLU) producing L2-normalized z_f.
      - Direct ECFP4 Head: Predicts 2048-bit probabilities p_hat for continuous Tanimoto evaluation.
    """

    def __init__(
        self,
        spec_bins: int = 2000,
        fp_bits: int = 2048,
        d_embed: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.spec_encoder = nn.Sequential(
            nn.Linear(spec_bins, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, d_embed),
        )

        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_bits, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, d_embed),
        )

        # Auxiliary Fingerprint FFN head (Table 4 of Carrera et al.)
        self.ecfp4_head = nn.Sequential(
            nn.Linear(d_embed, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, fp_bits),
        )

    def encode_spectrum(self, binned_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_z_s = self.spec_encoder(binned_spec)
        z_s = F.normalize(raw_z_s, p=2, dim=-1)
        fp_logits = self.ecfp4_head(raw_z_s)
        return z_s, fp_logits

    def encode_fingerprint(self, ecfp4: torch.Tensor) -> torch.Tensor:
        raw_z_f = self.fp_encoder(ecfp4)
        return F.normalize(raw_z_f, p=2, dim=-1)

    def forward(
        self, binned_spec: torch.Tensor, ecfp4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_s, fp_logits = self.encode_spectrum(binned_spec)
        z_f = self.encode_fingerprint(ecfp4)
        return z_s, z_f, fp_logits


class SiameseContrastiveTanimotoLoss(nn.Module):
    """
    Combines Cosine Contrastive Metric Loss on (z_s, z_f) pairs with
    differentiable Tanimoto + Binary Cross-Entropy loss on the 2048-bit ECFP4 head.
    """

    def __init__(self, margin: float = 0.25, tau: float = 0.07, fp_weight: float = 0.5):
        super().__init__()
        self.margin = margin
        self.tau = tau
        self.fp_weight = fp_weight

    @staticmethod
    def continuous_tanimoto_similarity(pred_probs: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
        # pred_probs: (B, 2048), target_bits: (B, 2048)
        intersection = torch.sum(pred_probs * target_bits, dim=-1)
        union = (
            torch.sum(pred_probs.pow(2), dim=-1)
            + torch.sum(target_bits.pow(2), dim=-1)
            - intersection
            + 1e-6
        )
        return intersection / union

    def forward(
        self,
        z_s: torch.Tensor,
        z_f: torch.Tensor,
        fp_logits: torch.Tensor,
        target_ecfp4: torch.Tensor,
    ) -> torch.Tensor:
        B = z_s.size(0)
        labels = torch.arange(B, device=z_s.device)

        # 1. Symmetric InfoNCE + Margin Cosine Contrastive Loss in 256-d Siamese space
        sim_matrix = torch.matmul(z_s, z_f.T)
        infonce_loss = 0.5 * (
            F.cross_entropy(sim_matrix / self.tau, labels)
            + F.cross_entropy(sim_matrix.T / self.tau, labels)
        )

        pos_sim = torch.diag(sim_matrix)
        neg_mask = ~torch.eye(B, dtype=torch.bool, device=z_s.device)
        hardest_neg = sim_matrix.masked_fill(~neg_mask, -1.0).max(dim=-1).values
        margin_loss = F.relu(hardest_neg - pos_sim + self.margin).mean()

        # 2. Direct 2048-bit ECFP4 BCE + (1 - Tanimoto) loss
        bce_loss = F.binary_cross_entropy_with_logits(fp_logits, target_ecfp4)
        pred_probs = torch.sigmoid(fp_logits)
        tanimoto_loss = 1.0 - self.continuous_tanimoto_similarity(pred_probs, target_ecfp4).mean()

        return infonce_loss + margin_loss + self.fp_weight * (bce_loss + tanimoto_loss)
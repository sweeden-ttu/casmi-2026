# [Colab Ensemble Cell 3] Mass2SMILES Temporal Convolutional Network (TCN) + 71-D Functional Group Head
class DilatedTCNBlock(nn.Module):
    """
    Residual Temporal Convolutional Block with kernel_size=8 and 256 filters
    (Elser et al., Mass2SMILES Methods).
    """

    def __init__(
        self,
        channels: int = 256,
        kernel_size: int = 8,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.padding = padding

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        out = F.gelu(self.norm1(self._chomp(self.conv1(x))))
        out = self.dropout(out)
        out = F.gelu(self.norm2(self._chomp(self.conv2(out))))
        out = self.dropout(out)
        return F.gelu(x + out)


class Mass2SMILESFunctionalGroupHead(nn.Module):
    """
    Consumes multi-spectrum sinusoidal sequence states H^s_mol (B, P, 256) from the
    GLMR spectral encoder, passes them through a dilated TCN stack (kernel_size=8,
    256 filters), and outputs:
      1. fg_71_pred (B, 71): 60 RDKit functional groups + 8 heteroatoms + 3 sugar/ring counts
      2. cddd_512_pred (B, 512): Continuous structural descriptor embedding
    """

    def __init__(
        self,
        d_model: int = 256,
        kernel_size: int = 8,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
        out_fg_dim: int = 71,
        out_desc_dim: int = 512,
    ):
        super().__init__()
        self.tcn_stack = nn.ModuleList(
            [
                DilatedTCNBlock(channels=d_model, kernel_size=kernel_size, dilation=d)
                for d in dilations
            ]
        )
        self.attn_pool = nn.Linear(d_model, 1)

        # 2x2 Dense layers producing outputs of shape 71 and 512 (Mass2SMILES Figure 1)
        self.fg_branch = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, out_fg_dim),
            nn.Softplus(),  # Non-negative log1p functional group & atom counts
        )
        self.desc_branch = nn.Sequential(
            nn.Linear(d_model, out_desc_dim),
            nn.LayerNorm(out_desc_dim),
            nn.GELU(),
            nn.Linear(out_desc_dim, out_desc_dim),
        )

    def forward(self, h_s_mol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_s_mol: (B, P, 256) -> transpose to (B, 256, P) for 1D temporal convolutions
        x = h_s_mol.transpose(1, 2)
        for block in self.tcn_stack:
            x = block(x)
        x_seq = x.transpose(1, 2)  # (B, P, 256)

        weights = F.softmax(self.attn_pool(x_seq).squeeze(-1), dim=-1)  # (B, P)
        pooled = torch.sum(x_seq * weights.unsqueeze(-1), dim=1)         # (B, 256)

        fg_71_pred = self.fg_branch(pooled)
        cddd_512_pred = F.normalize(self.desc_branch(pooled), p=2, dim=-1)
        return fg_71_pred, cddd_512_pred


def mass2smiles_fg_loss(pred_fg: torch.Tensor, target_fg: torch.Tensor) -> torch.Tensor:
    """
    Joint MAE + MSE loss used to train Mass2SMILES functional group predictions
    (Elser et al., Figure 2C & 2D).
    """
    mae = F.l1_loss(pred_fg, target_fg)
    mse = F.mse_loss(pred_fg, target_fg)
    return mae + mse
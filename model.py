import math
import torch
import torch.nn as nn


class MultiScaleCNN(nn.Module):
    def __init__(self, n_channels, model_dim, dropout=0.3):
        super().__init__()

        branch_dim = model_dim // 4

        self.branch_small = nn.Sequential(
            nn.Conv1d(n_channels, branch_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )

        self.branch_medium = nn.Sequential(
            nn.Conv1d(n_channels, branch_dim, kernel_size=25, padding=12),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=25, padding=12),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )

        self.branch_large = nn.Sequential(
            nn.Conv1d(n_channels, branch_dim, kernel_size=50, padding=25),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=50, padding=25),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )

        self.branch_xlarge = nn.Sequential(
            nn.Conv1d(n_channels, branch_dim, kernel_size=100, padding=50),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.Conv1d(branch_dim, branch_dim, kernel_size=100, padding=50),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.GELU(),
            nn.MaxPool1d(5),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)

        small = self.branch_small(x)
        medium = self.branch_medium(x)
        large = self.branch_large(x)
        xlarge = self.branch_xlarge(x)

        multi = torch.cat([small, medium, large, xlarge], dim=1)

        out = self.fusion(multi)

        return out.permute(0, 2, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SleepTransformer(nn.Module):
    def __init__(
        self,
        n_channels,
        seq_len,
        n_classes,
        model_dim=128,
        num_heads=8,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.model_dim = model_dim

        self.cnn = MultiScaleCNN(n_channels, model_dim, dropout)

        self.pos_encoder = PositionalEncoding(
            model_dim,
            max_len=1000,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(model_dim, 1),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.cnn(x)

        features = self.pos_encoder(features)

        features = self.transformer(features)

        attn_weights = self.attention_pool(features)

        attn_pooled = (features * attn_weights).sum(dim=1)

        mean_pooled = features.mean(dim=1)

        pooled = torch.cat([attn_pooled, mean_pooled], dim=1)

        return self.classifier(pooled)
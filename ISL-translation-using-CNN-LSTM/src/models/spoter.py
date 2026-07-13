import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SPOTER(nn.Module):
    """
    Sign POse-based TransformER.
    CLS token + Transformer Encoder for word-level sign recognition.

    Input:  (B, T, input_dim)  — T=64, input_dim=543
    Output: (B, num_classes)
    """

    def __init__(
        self,
        input_dim=543,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=512,
        num_classes=263,
        dropout=0.1,
        max_seq_len=64
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T, input_dim)
        B = x.size(0)

        x = self.input_proj(x)                              # (B, T, d_model)

        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                      # (B, T+1, d_model)

        x = self.pos_enc(x)                                 # (B, T+1, d_model)

        # Prepend False to mask for CLS token (never masked)
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T+1, d_model)

        cls_out = x[:, 0]                                   # (B, d_model)
        return self.classifier(cls_out)                     # (B, num_classes)

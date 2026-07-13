# Model Architectures

Three models in order of implementation complexity. Build and validate in order.

---

## `src/dataset.py` — Shared Dataset Class

```python
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from augment import augment as apply_augment


class ISLDataset(Dataset):
    def __init__(self, npy_dir, split_file, mean_path, std_path, augment=False):
        """
        Args:
            npy_dir: Directory containing .npy keypoint files
            split_file: Path to split txt (format: "filename_labelid.npy" per line
                        OR "relative/video/path labelid" — see note below)
            mean_path: Path to mean.npy for normalization
            std_path: Path to std.npy for normalization
            augment: Whether to apply augmentation
        """
        self.npy_dir = npy_dir
        self.augment = augment
        self.mean = np.load(mean_path)  # (543,)
        self.std = np.load(std_path)    # (543,)

        # Build file list from directory (naming convention: vidname_labelid.npy)
        self.samples = []
        for fname in os.listdir(npy_dir):
            if not fname.endswith('.npy'):
                continue
            label_id = int(fname.rsplit('_', 1)[-1].replace('.npy', ''))
            self.samples.append((fname, label_id))

        # If split_file given, filter to only those files
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                allowed = set(line.strip() for line in f)
            self.samples = [(f, l) for f, l in self.samples if f in allowed]

        assert len(self.samples) > 0, f"No samples found in {npy_dir}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        seq = np.load(os.path.join(self.npy_dir, fname))  # (64, 543)

        if self.augment:
            seq = apply_augment(seq)

        seq = (seq - self.mean) / self.std  # z-score normalize
        return torch.FloatTensor(seq), torch.tensor(label, dtype=torch.long)
```

---

## Model 1: BiLSTM + Attention Baseline

**File: `src/models/bilstm.py`**

Train this first. Fast to converge, easy to debug. Expected accuracy on INCLUDE: ~60-70%.

```python
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with soft attention pooling for sign recognition.
    Input:  (B, T, input_dim)
    Output: (B, num_classes)
    """

    def __init__(self, input_dim=543, hidden=256, num_layers=2,
                 num_classes=263, dropout=0.3):
        super().__init__()

        # Input projection (reduce dimensionality before LSTM)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention: score each timestep
        self.attention = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 543)
        x = self.input_proj(x)                        # (B, T, 256)
        out, _ = self.lstm(x)                         # (B, T, hidden*2)

        scores = self.attention(out)                  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)        # (B, T, 1)
        context = (weights * out).sum(dim=1)          # (B, hidden*2)

        return self.classifier(context)               # (B, num_classes)
```

---

## Model 2: SPOTER (Primary Model)

**File: `src/models/spoter.py`**

Transformer encoder with CLS token. Purpose-built for pose-based sign recognition. Expected accuracy on INCLUDE: ~70-80%.

```python
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
```

---

## Model 3: SPOTER + Encoder-Decoder (Sentence-Level Extension)

**File: `src/models/slt_model.py`**

Only implement after word-level SPOTER works. Uses pretrained SPOTER encoder + autoregressive decoder.

```python
import torch
import torch.nn as nn
from spoter import SPOTER


class SLTModel(nn.Module):
    """
    Sign Language Translation model.
    Encoder: Pretrained SPOTER encoder (load weights, remove classifier head)
    Decoder: Transformer decoder → text tokens

    Input:  video keypoints (B, T, 543) + target tokens (B, S)
    Output: logits over vocabulary (B, S, vocab_size)
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        max_src_len=64,
        max_tgt_len=32,
        pretrained_encoder_path=None
    ):
        super().__init__()

        # Encoder: reuse SPOTER minus its classifier
        self.encoder_model = SPOTER(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            num_classes=1,  # dummy, we won't use the head
            dropout=dropout
        )
        if pretrained_encoder_path:
            state = torch.load(pretrained_encoder_path, map_location='cpu')
            # Load encoder weights only, skip classifier
            encoder_state = {k: v for k, v in state.items()
                             if not k.startswith('classifier')}
            self.encoder_model.load_state_dict(encoder_state, strict=False)

        # Token embedding for decoder
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_pos_enc = nn.Embedding(max_tgt_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.output_proj = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        # src: (B, T, 543) → (B, T+1, d_model)
        B = src.size(0)
        x = self.encoder_model.input_proj(src)
        cls = self.encoder_model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder_model.pos_enc(x)
        return self.encoder_model.encoder(x)  # (B, T+1, d_model)

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: (B, S) token ids
        S = tgt.size(1)
        positions = torch.arange(S, device=tgt.device).unsqueeze(0)
        x = self.tgt_embedding(tgt) + self.tgt_pos_enc(positions)
        out = self.decoder(x, memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask)
        return self.output_proj(out)  # (B, S, vocab_size)

    def forward(self, src, tgt):
        memory = self.encode(src)
        S = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=src.device)
        return self.decode(tgt, memory, tgt_mask=tgt_mask)
```

---

## Model Selection Guide

| Scenario | Use |
|---|---|
| First run, debugging pipeline | BiLSTM |
| Primary experiment, paper baseline | SPOTER |
| Low data, need strong features | WLASL-pretrained SPOTER fine-tuned |
| Sentence-level output needed | SLTModel |

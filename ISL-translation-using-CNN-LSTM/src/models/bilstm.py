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

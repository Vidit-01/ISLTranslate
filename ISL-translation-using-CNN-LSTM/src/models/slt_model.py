import torch
import torch.nn as nn
from models.spoter import SPOTER


class SLTModel(nn.Module):
    """
    Sign Language Translation model.
    Encoder: Pretrained SPOTER encoder (load weights, remove classifier head)
    Decoder: Transformer decoder -> text tokens

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
        # src: (B, T, 543) -> (B, T+1, d_model)
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

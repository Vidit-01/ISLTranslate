# Extension: Sentence-Level Translation (SLT)

**Prerequisite:** Word-level SPOTER working with ≥ 70% Top-1 accuracy.

---

## Overview

SLT is a sequence-to-sequence problem:
```
Video of continuous signing → English sentence
```

This requires:
1. A visual encoder that processes signing sequences (reuse SPOTER encoder)
2. A text decoder that autoregressively generates output tokens
3. A sentence-level dataset (ISL-CSLRT or PHOENIX-2014T for pretraining)

---

## Dataset for SLT

### PHOENIX-2014T (Pretrain)
- German Sign Language (DGS), weather broadcast domain
- 8257 train / 519 dev / 642 test sentence pairs
- Download: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/
- Standard benchmark: report BLEU-4 on test set
- Use for architecture validation before switching to ISL

### ISL-CSLRT (Fine-tune)
- Indian Sign Language, sentence-level
- Smaller than PHOENIX, domain-specific
- Request via IIT Delhi research group

---

## Tokenizer

Use a simple word-level tokenizer on gloss/text vocabulary, or use a pretrained BPE tokenizer:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def build_tokenizer(sentences, save_path, vocab_size=1000):
    """Build BPE tokenizer on translation corpus."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    )
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
```

---

## SLT Dataset Class

```python
class SLTDataset(Dataset):
    def __init__(self, npy_dir, annotation_file, tokenizer, mean, std,
                 max_src_len=64, max_tgt_len=32, augment=False):
        """
        annotation_file: JSON list of {"video_id": str, "translation": str}
        """
        import json
        self.npy_dir = npy_dir
        self.tokenizer = tokenizer
        self.mean = mean
        self.std = std
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.augment = augment

        with open(annotation_file) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vid_id = sample['video_id']
        translation = sample['translation']

        # Load keypoints
        npy_path = os.path.join(self.npy_dir, f"{vid_id}.npy")
        src = np.load(npy_path)
        if self.augment:
            src = apply_augment(src)
        src = (src - self.mean) / self.std

        # Tokenize target
        encoded = self.tokenizer.encode(translation)
        tgt_ids = [BOS_ID] + encoded.ids[:self.max_tgt_len - 2] + [EOS_ID]
        tgt_in = tgt_ids[:-1]   # decoder input (BOS ... last-token)
        tgt_out = tgt_ids[1:]   # decoder target (first-token ... EOS)

        return (
            torch.FloatTensor(src),
            torch.tensor(tgt_in, dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long)
        )


def slt_collate_fn(batch):
    """Pad variable-length target sequences."""
    srcs, tgt_ins, tgt_outs = zip(*batch)
    srcs = torch.stack(srcs)
    tgt_ins = torch.nn.utils.rnn.pad_sequence(tgt_ins, batch_first=True, padding_value=PAD_ID)
    tgt_outs = torch.nn.utils.rnn.pad_sequence(tgt_outs, batch_first=True, padding_value=PAD_ID)
    return srcs, tgt_ins, tgt_outs
```

---

## SLT Training Loop

```python
def train_slt_epoch(model, loader, optimizer, scaler, device):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    total_loss, total_tokens = 0.0, 0

    for src, tgt_in, tgt_out in tqdm(loader, desc="SLT train"):
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(src, tgt_in)  # (B, S, vocab_size)
            B, S, V = logits.shape
            loss = criterion(logits.reshape(B * S, V), tgt_out.reshape(B * S))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        mask = tgt_out != PAD_ID
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()

    return total_loss / total_tokens
```

---

## BLEU Evaluation

```python
from sacrebleu.metrics import BLEU

def evaluate_bleu(model, loader, tokenizer, device, max_tgt_len=32):
    model.eval()
    hypotheses, references = [], []

    for src, tgt_in, tgt_out in tqdm(loader, desc="BLEU eval"):
        src = src.to(device)
        B = src.size(0)

        # Greedy decoding
        memory = model.encode(src)
        tgt = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)

        for _ in range(max_tgt_len):
            S = tgt.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
            logits = model.decode(tgt, memory, tgt_mask=tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_tok], dim=1)
            if (next_tok == EOS_ID).all():
                break

        for i in range(B):
            ids = tgt[i, 1:].tolist()  # skip BOS
            if EOS_ID in ids:
                ids = ids[:ids.index(EOS_ID)]
            hyp = tokenizer.decode(ids)
            hypotheses.append(hyp)

        # Decode reference
        for i in range(B):
            ref_ids = tgt_out[i].tolist()
            if EOS_ID in ref_ids:
                ref_ids = ref_ids[:ref_ids.index(EOS_ID)]
            ref = tokenizer.decode(ref_ids)
            references.append(ref)

    bleu = BLEU()
    score = bleu.corpus_score(hypotheses, [references])
    print(f"BLEU-4: {score}")
    return score
```

---

## Transfer Learning Strategy

Best approach given scarce ISL data:

```
1. Pretrain SPOTER (word-level) on WLASL (21k videos, 2000 classes)
2. Pretrain SLTModel on PHOENIX-2014T (encoder-decoder, sentence-level)
3. Fine-tune on ISL-CSLRT with low lr (1e-5), freeze encoder for first 10 epochs
4. Unfreeze encoder, train end-to-end with lr=5e-6
```

Expected BLEU-4 on PHOENIX-2014T: ~20-25 (comparable to published baselines).
ISL-CSLRT will be lower due to smaller training set.

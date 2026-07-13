import os
import yaml
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import ISLDataset
from models.bilstm import BiLSTMClassifier
from models.spoter import SPOTER


def build_model(cfg):
    if cfg['model'] == 'bilstm':
        return BiLSTMClassifier(
            input_dim=cfg['input_dim'],
            hidden=cfg['hidden'],
            num_layers=cfg['num_layers'],
            num_classes=cfg['num_classes'],
            dropout=cfg['dropout']
        )
    elif cfg['model'] == 'spoter':
        return SPOTER(
            input_dim=cfg['input_dim'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_encoder_layers=cfg['num_encoder_layers'],
            dim_feedforward=cfg['dim_feedforward'],
            num_classes=cfg['num_classes'],
            dropout=cfg['dropout']
        )
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")


def get_scheduler(optimizer, cfg, steps_per_epoch):
    train_cfg = cfg['train']
    total_steps = train_cfg['epochs'] * steps_per_epoch
    warmup_steps = train_cfg.get('warmup_epochs', 0) * steps_per_epoch

    if train_cfg['scheduler'] == 'cosine':
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=train_cfg['lr'],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
            anneal_strategy='cos'
        )
        return scheduler, 'step'  # call every batch
    elif train_cfg['scheduler'] == 'step':
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=[60, 100], gamma=0.1)
        return scheduler, 'epoch'  # call every epoch
    else:
        raise ValueError(f"Unknown scheduler: {train_cfg['scheduler']}")


def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler,
                    scheduler_mode, grad_clip, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler_mode == 'step':
            scheduler.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, top5_correct, total = 0.0, 0, 0, 0

    for x, y in tqdm(loader, desc="  eval", leave=False):
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()

        # Top-5
        top5 = logits.topk(5, dim=1).indices
        top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total, top5_correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }, path)


def train(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_cfg = cfg['data']
    train_cfg = cfg['train']
    ckpt_cfg = cfg['checkpoint']

    splits_dir = data_cfg['splits_dir']
    mean_path = os.path.join(splits_dir, 'mean.npy')
    std_path = os.path.join(splits_dir, 'std.npy')

    train_ds = ISLDataset(
        npy_dir=data_cfg['npy_dir'],
        split_file=os.path.join(splits_dir, 'train.txt'),
        mean_path=mean_path,
        std_path=std_path,
        augment=data_cfg['augment_train']
    )
    val_ds = ISLDataset(
        npy_dir=data_cfg['npy_dir'],
        split_file=os.path.join(splits_dir, 'val.txt'),
        mean_path=mean_path,
        std_path=std_path,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=2, pin_memory=True,
                            persistent_workers=True)

    model = build_model(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg['label_smoothing'])
    scaler = GradScaler()
    scheduler, scheduler_mode = get_scheduler(optimizer, cfg, len(train_loader))

    best_val_acc = 0.0
    log = []

    for epoch in range(1, train_cfg['epochs'] + 1):
        print(f"Epoch {epoch}/{train_cfg['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            scheduler, scheduler_mode, train_cfg['grad_clip'], device
        )
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)

        if scheduler_mode == 'epoch':
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} top5={val_top5:.4f} | "
              f"lr={lr_now:.2e}")

        log.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_top5': val_top5
        })

        # Save checkpoint
        if epoch % ckpt_cfg['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, val_acc,
                            os.path.join(ckpt_cfg['dir'], f'epoch_{epoch}.pt'))

        if ckpt_cfg['save_best'] and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc,
                            os.path.join(ckpt_cfg['dir'], 'best.pt'))
            print(f"  *** New best: {best_val_acc:.4f} ***")

        # Save log
        log_path = os.path.join(ckpt_cfg['dir'], 'log.json')
        os.makedirs(ckpt_cfg['dir'], exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)

    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    train(args.config)

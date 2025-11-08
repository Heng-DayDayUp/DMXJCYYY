# train.py
import os
# Workaround for Windows OpenMP duplicate lib issue (libiomp5md.dll)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')

import argparse
import random
from pathlib import Path
import time
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data import download_tiny_shakespeare, CharDataset
from model import TransformerSeq2Seq, TransformerEncoder, count_parameters
from utils import save_checkpoint, plot_train_curve, save_vocab, save_json

# non-interactive matplotlib
import matplotlib
matplotlib.use('Agg')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_scheduler(optimizer, warmup_steps=1000):
    def lr_lambda(step):
        if step == 0:
            return 1.0
        return min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5))
    return LambdaLR(optimizer, lr_lambda)


def create_masks_for_seq2seq(src, tgt, pad_idx=None):
    # For char-level tiny dataset, assume no padding; if padding used, modify accordingly.
    device = src.device
    # src_mask not used here (no padding)
    src_mask = None
    tgt_len = tgt.size(1)
    tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).unsqueeze(0).unsqueeze(0)  # (1,1,tgt,tgt)
    memory_mask = None
    return src_mask, tgt_mask, memory_mask


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset download
    data_path = download_tiny_shakespeare(args.data)
    ds = CharDataset(data_path, seq_len=args.seq_len, mode=('lm' if args.task == 'lm' else 'seq2seq'))
    vocab_size = ds.vocab_size
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=ds.collate_fn, drop_last=True)

    # Model selection: for seq2seq we use TransformerSeq2Seq with same vocab for src/tgt by default.
    if args.task == 'seq2seq':
        model = TransformerSeq2Seq(src_vocab=vocab_size, tgt_vocab=vocab_size,
                                   d_model=args.d_model, num_layers=args.layers, num_heads=args.heads,
                                   d_ff=args.d_ff, dropout=args.dropout, max_len=args.max_len,
                                   use_pos=args.use_pos_encoding, use_relative=args.relative_pos,
                                   max_rel_pos=args.max_rel, use_residual=(not args.no_residual))
    else:  # 'lm'
        model = TransformerEncoder(vocab_size=vocab_size, d_model=args.d_model, num_layers=args.layers,
                                   num_heads=args.heads, d_ff=args.d_ff, dropout=args.dropout, max_len=args.max_len,
                                   use_pos=args.use_pos_encoding, use_relative=args.relative_pos, max_rel_pos=args.max_rel,
                                   use_residual=(not args.no_residual))

    model.to(device)
    print('Params:', count_parameters(model))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, warmup_steps=args.warmup)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    model_dir = Path(args.save)
    model_dir.mkdir(parents=True, exist_ok=True)

    # save training config for reproducibility
    config_path = model_dir / 'train_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            if args.task == 'seq2seq':
                # For decoder input, provide <bos> or shifted target; here we use y_input = [<pad?>, y[:, :-1]]
                # Simpler: shift right by one and use last token as target. For char-level, treat y as target and y_input as previous tokens.
                y_input = torch.zeros_like(y)
                y_input[:, 1:] = y[:, :-1]
                # y_input[:,0] stays 0 (could be special BOS if using special token)
                src_mask, tgt_mask, memory_mask = create_masks_for_seq2seq(x, y_input)
                logits, self_attns, enc_attns = model(x, y_input, src_mask, tgt_mask, memory_mask)
                # logits: (batch, seq, vocab)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                logits, _, _ = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            epoch_loss += loss.item()
            global_step += 1

            if (i % args.log_interval) == 0:
                print(f'Epoch {epoch} Step {i} Loss {loss.item():.4f}')

        end = time.time()
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f'Epoch {epoch} finished in {end-start:.1f}s average loss {avg_loss:.4f}')
        # save checkpoint each epoch
        ckpt = model_dir / f'model_epoch{epoch}.pt'
        save_checkpoint(model, optimizer, scheduler, epoch, ckpt, extra={'vocab': ds.chars})

    # plot
    plot_train_curve(train_losses, model_dir / 'train_loss.png')
    # save vocab
    save_vocab(ds.chars, model_dir / 'vocab.json')
    print('Training finished. Artifacts in', model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['lm', 'seq2seq'], default='seq2seq', help='Task type')
    parser.add_argument('--data', type=str, default='data/tiny_shakespeare.txt')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save', type=str, default='results')
    parser.add_argument('--max_len', type=int, default=5000)
    parser.add_argument('--use_pos_encoding', action='store_true', help='Use sinusoidal positional encoding')
    parser.add_argument('--relative_pos', action='store_true', help='Use relative positional bias')
    parser.add_argument('--max_rel', type=int, default=128, help='Max relative position (for relative bias)')
    parser.add_argument('--no_residual', action='store_true', help='Disable residual connections for ablation')
    args = parser.parse_args()
    train(args)

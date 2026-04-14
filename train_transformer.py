import random
import string
import numpy as np
import torch
from common import _token_kind, build_kind_lookup, load_text_lines
from settings import train_file, val_file, vocab_file

val_size = 300
embedding_dim = 128
kind_embedding_dim = 16
space_embedding_dim = 8
d_model = 256
nhead = 8
num_layers = 4
dim_feedforward = 768
head_dropout = 0.1
dropout = 0.1
learning_rate = 3e-4
weight_decay = 1e-2
num_epochs = 100
warmup_epochs = 4
batch_size = 32
max_length = 400
pad_token = "<pad>"
lines_per_epoch = 1000
scheduler_patience = 4
scheduler_factor = 0.5
validation_ema_beta = 0.5
threshold_grid = np.linspace(0.05, 0.95, 19)
space_noise_add_prob = 0.08
space_noise_remove_prob = 0.05
num_threads = 12
pad_id = None
unk_id = None

class WhitespaceCorrector(torch.nn.Module):
    def __init__(self, char_vocab_size, kind_lookup, head_bias_init):
        super().__init__()
        self.char_embedding = torch.nn.Embedding(char_vocab_size, embedding_dim, padding_idx=pad_id)
        self.kind_embedding = torch.nn.Embedding(7, kind_embedding_dim)
        self.space_embedding = torch.nn.Embedding(2, space_embedding_dim)
        self.input_proj = torch.nn.Linear(embedding_dim + kind_embedding_dim + space_embedding_dim, d_model)
        self.pos_embedding = torch.nn.Embedding(max_length, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=torch.nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head_drop = torch.nn.Dropout(head_dropout)
        self.head = torch.nn.Linear(d_model, 1)
        torch.nn.init.constant_(self.head.bias, head_bias_init)
        self.register_buffer("kind_lookup", kind_lookup, persistent=False)

    def forward(self, x, space_flags):
        B, L = x.shape
        pad_mask = (x == pad_id)
        kind_ids = self.kind_lookup[x]
        char_emb = self.char_embedding(x)
        kind_emb = self.kind_embedding(kind_ids)
        space_emb = self.space_embedding(space_flags)
        pos_emb = self.pos_embedding(torch.arange(L, device=x.device).unsqueeze(0))
        out = self.input_proj(torch.cat([char_emb, kind_emb, space_emb], dim=-1)) + pos_emb
        out = self.transformer(out, src_key_padding_mask=pad_mask)
        out = self.head_drop(out)
        return self.head(out).squeeze(-1)

def prepare_batch(batch_lines, vocab, apply_noise):
    inputs = []
    space_flags_list = []
    labels = []
    for line in batch_lines:
        chars = []
        flags = []
        preceded_by_space = False
        for ch in line:
            if ch == " ":
                preceded_by_space = True
                continue
            chars.append(vocab.get(ch, unk_id))
            flags.append(1 if preceded_by_space else 0)
            preceded_by_space = False
        targets = [0.0] * len(chars)
        for j in range(len(targets) - 1):
            if flags[j + 1] == 1:
                targets[j] = 1.0
        if apply_noise:
            for j in range(len(flags)):
                if flags[j] == 0 and random.random() < space_noise_add_prob:
                    flags[j] = 1
                elif flags[j] == 1 and random.random() < space_noise_remove_prob:
                    flags[j] = 0
        if flags and flags[0] == 1:
            flags[0] = 0
        chars = chars[:max_length]
        flags = flags[:max_length]
        targets = targets[:max_length]
        if len(chars) == 0:
            continue
        inputs.append(torch.tensor(chars, dtype=torch.long))
        space_flags_list.append(torch.tensor(flags, dtype=torch.long))
        labels.append(torch.tensor(targets, dtype=torch.float32))
    if not inputs:
        return None, None, None
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    flags_padded = torch.nn.utils.rnn.pad_sequence(space_flags_list, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)
    return input_padded, flags_padded, labels_padded

def _counts_from_predictions(preds, labels):
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    return tp, fp, fn

def _metrics_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def estimate_positive_rate(lines, vocab, n_samples=2000):
    sampled = random.sample(lines, min(n_samples, len(lines)))
    total_tokens = 0
    total_positives = 0
    for line in sampled:
        flags = []
        preceded_by_space = False
        for ch in line:
            if ch == " ":
                preceded_by_space = True
                continue
            flags.append(1 if preceded_by_space else 0)
            preceded_by_space = False
        targets = [0.0] * len(flags)
        for j in range(len(targets) - 1):
            if flags[j + 1] == 1:
                targets[j] = 1.0
        total_tokens += len(targets)
        total_positives += sum(targets)
    if total_tokens == 0:
        return 0.5
    return total_positives / total_tokens

def train_epoch(model, optimizer, train_lines, vocab, pos_weight):
    model.train()
    indices = random.sample(range(len(train_lines)), min(lines_per_epoch, len(train_lines)))
    total_loss = 0.0
    total_tokens = 0.0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_lines = [train_lines[i] for i in batch_idx]
        input_padded, flags_padded, labels_padded = prepare_batch(batch_lines, vocab, apply_noise=True)
        if input_padded is None:
            continue
        logits = model(input_padded, flags_padded)
        mask = (input_padded != pad_id).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels_padded, pos_weight=pos_weight, reduction="none"
        )
        loss_sum = (loss * mask).sum()
        token_count = mask.sum()
        if token_count > 0:
            loss_mean = loss_sum / token_count
            optimizer.zero_grad(set_to_none=True)
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss_sum.item()
            total_tokens += token_count.item()
    return total_loss / total_tokens if total_tokens > 0 else 0.0

def evaluate(model, vocab, val_lines):
    model.eval()
    sampled_lines = random.sample(val_lines, min(val_size, len(val_lines)))
    all_probs = []
    all_labels = []
    with torch.inference_mode():
        for start in range(0, len(sampled_lines), batch_size):
            batch_lines = sampled_lines[start:start + batch_size]
            input_padded, flags_padded, labels_padded = prepare_batch(batch_lines, vocab, apply_noise=True)
            if input_padded is None:
                continue
            logits = model(input_padded, flags_padded)
            mask = (input_padded != pad_id)
            probs = torch.sigmoid(logits)[mask].detach().cpu().numpy()
            labels = labels_padded[mask].detach().cpu().numpy().astype(np.int32)
            if probs.size == 0:
                continue
            all_probs.append(probs)
            all_labels.append(labels)
    if not all_probs:
        return 0.0, 0.0, 0.5, 0.0, 0.0, 0.0
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    raw_preds = (probs >= 0.5).astype(np.int32)
    tp, fp, fn = _counts_from_predictions(raw_preds, labels)
    raw_precision, raw_recall, raw_f1 = _metrics_from_counts(tp, fp, fn)
    best_f1, best_threshold, best_precision, best_recall = -1.0, 0.5, 0.0, 0.0
    for thr in threshold_grid:
        preds = (probs >= thr).astype(np.int32)
        tp, fp, fn = _counts_from_predictions(preds, labels)
        precision, recall, f1 = _metrics_from_counts(tp, fp, fn)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
            best_precision = precision
            best_recall = recall
    predicted_positive_rate = float(raw_preds.mean()) if raw_preds.size else 0.0
    return raw_f1, best_f1, best_threshold, best_precision, best_recall, predicted_positive_rate

def main():
    global pad_id, unk_id
    torch.set_num_threads(num_threads)
    checkpoint = torch.load(vocab_file, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    pad_id = vocab[pad_token]
    unk_id = vocab.get("<unk>", pad_id)
    train_lines = load_text_lines(train_file)
    val_lines = load_text_lines(val_file)
    vocab_size = max(vocab.values()) + 1 if isinstance(vocab, dict) else len(vocab)
    kind_lookup = build_kind_lookup(vocab, pad_token)
    pos_rate = estimate_positive_rate(train_lines, vocab)
    print(f"Estimated positive rate: {pos_rate:.4f}")
    head_bias_init = np.log(pos_rate / (1.0 - pos_rate)) if 0.0 < pos_rate < 1.0 else 0.0
    print(f"Head bias init: {head_bias_init:.4f}")
    model = WhitespaceCorrector(vocab_size, kind_lookup, head_bias_init)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pos_weight = torch.tensor([(1.0 - pos_rate) / pos_rate], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience)
    best_f1 = 0.0
    ema_val = None
    best_threshold = 0.5
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = learning_rate * warmup_factor
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{num_epochs}, lr: {current_lr:.2e}")
        train_loss = train_epoch(model, optimizer, train_lines, vocab, pos_weight)
        raw_f1, best_epoch_f1, epoch_threshold, precision, recall, positive_rate = evaluate(model, vocab, val_lines)
        ema_val = best_epoch_f1 if ema_val is None else validation_ema_beta * ema_val + (1.0 - validation_ema_beta) * best_epoch_f1
        print(f"Loss: {train_loss:.3f}, F1@0.5: {raw_f1:.2f}, best F1: {best_epoch_f1:.2f}, best threshold: {epoch_threshold:.2f}, precision: {precision:.2f}, recall: {recall:.2f}")
        if epoch >= warmup_epochs:
            scheduler.step(ema_val)
        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_threshold = epoch_threshold
            torch.save({"model": model.state_dict(), "vocab": vocab, "threshold": best_threshold, "best_f1": best_f1}, "whitespace.pth")
            print(f"New best: {best_f1:.2f}")
    torch.save({"model": model.state_dict(), "vocab": vocab, "threshold": best_threshold, "best_f1": best_f1}, "whitespace.pth")
    print(f"Training finished. Best validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()

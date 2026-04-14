import random
import string
import numpy as np
import torch
from common import _token_kind, build_kind_lookup, load_text_lines
from settings import train_file, val_file, vocab_file, model_file_name

val_size = 300
embedding_dim = 128
kind_embedding_dim = 16
hidden_dim = 256
num_blocks = 6
dropout = 0.1
learning_rate = 3e-4
weight_decay = 1e-2
num_epochs = 40
batch_size = 32
max_length = 600
pad_token = "<pad>"
lines_per_epoch = 1000
scheduler_patience = 4
scheduler_factor = 0.5
validation_ema_beta = 0.8
threshold_grid = np.linspace(0.05, 0.95, 19)
space_injection_prob = 0.10
pad_id = None
unk_id = None

class ConvBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout_prob):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.norm = torch.nn.LayerNorm(channels)
        self.dw = torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad, groups=channels)
        self.pw = torch.nn.Conv1d(channels, channels, 1)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout_prob)
    def forward(self, x):
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.dw(y)
        y = self.pw(y)
        y = y.transpose(1, 2)
        y = self.act(y)
        y = self.drop(y)
        return x + y

class WhitespaceCorrector(torch.nn.Module):
    def __init__(self, char_vocab_size, kind_lookup):
        super().__init__()
        channels = hidden_dim
        self.char_embedding = torch.nn.Embedding(char_vocab_size, embedding_dim, padding_idx=pad_id)
        self.kind_embedding = torch.nn.Embedding(7, kind_embedding_dim)
        self.input_proj = torch.nn.Linear(embedding_dim + kind_embedding_dim, channels)
        self.blocks = torch.nn.ModuleList([ConvBlock(channels, 3, d, dropout) for d in [1, 2, 4, 8, 16, 32]])
        self.norm = torch.nn.LayerNorm(channels)
        self.head = torch.nn.Linear(channels, 1)
        self.register_buffer("kind_lookup", kind_lookup, persistent=False)
    def forward(self, x):
        kind_ids = self.kind_lookup[x]
        char_emb = self.char_embedding(x)
        kind_emb = self.kind_embedding(kind_ids)
        out = torch.cat([char_emb, kind_emb], dim=-1)
        out = self.input_proj(out)
        for block in self.blocks:
            out = block(out)
        out = self.norm(out)
        return self.head(out).squeeze(-1)

def prepare_augmented_batch(batch_lines, vocab):
    inputs = []
    labels = []
    space_id = vocab.get(" ", unk_id)
    for line in batch_lines:
        input_chars = []
        label_vals = []
        for ch in line:
            if random.random() < space_injection_prob:
                input_chars.append(space_id)
                label_vals.append(1.0)
            ch_id = vocab.get(ch, unk_id)
            input_chars.append(ch_id)
            label_vals.append(0.0)
        input_chars = input_chars[:max_length]
        label_vals = label_vals[:max_length]
        inputs.append(torch.tensor(input_chars, dtype=torch.long))
        labels.append(torch.tensor(label_vals, dtype=torch.float32))
    if not inputs: return None, None
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)
    return input_padded, labels_padded

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

def train_epoch(model, optimizer, train_lines, vocab):
    model.train()
    indices = random.sample(range(len(train_lines)), min(lines_per_epoch, len(train_lines)))
    total_loss = 0.0
    total_tokens = 0.0
    space_id = vocab.get(" ", unk_id)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_lines = [train_lines[i] for i in batch_idx]
        input_padded, labels_padded = prepare_augmented_batch(batch_lines, vocab)
        if input_padded is None: continue
        logits = model(input_padded)
        mask = (input_padded == space_id).float()
        pos_count = (labels_padded * mask).sum()
        neg_count = ((1.0 - labels_padded) * mask).sum()
        batch_pos_weight = (neg_count / pos_count) if pos_count > 0 else torch.tensor(1.0, device=logits.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_padded, pos_weight=batch_pos_weight, reduction="none")
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
    space_id = vocab.get(" ", unk_id)
    with torch.inference_mode():
        for start in range(0, len(sampled_lines), batch_size):
            batch_lines = sampled_lines[start:start + batch_size]
            input_padded, labels_padded = prepare_augmented_batch(batch_lines, vocab)
            if input_padded is None: continue
            logits = model(input_padded)
            mask = (input_padded == space_id)
            probs = torch.sigmoid(logits)[mask].detach().cpu().numpy()
            labels = labels_padded[mask].detach().cpu().numpy().astype(np.int32)
            if probs.size == 0: continue
            all_probs.append(probs)
            all_labels.append(labels)
    if not all_probs: return 0.0, 0.0, 0.5, 0.0, 0.0, 0.0
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
    torch.set_num_threads(max(1, torch.get_num_threads()))
    checkpoint = torch.load(vocab_file, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    pad_id = vocab[pad_token]
    unk_id = vocab.get("<unk>", pad_id)
    train_lines = load_text_lines(train_file)
    val_lines = load_text_lines(val_file)
    vocab_size = max(vocab.values()) + 1 if isinstance(vocab, dict) else len(vocab)
    kind_lookup = build_kind_lookup(vocab, pad_token)
    model = WhitespaceCorrector(vocab_size, kind_lookup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience)
    best_f1 = 0.0
    ema_val = None
    best_threshold = 0.5
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{num_epochs}  lr={current_lr:.2e}")
        train_loss = train_epoch(model, optimizer, train_lines, vocab)
        raw_f1, best_epoch_f1, epoch_threshold, precision, recall, positive_rate = evaluate(model, vocab, val_lines)
        ema_val = best_epoch_f1 if ema_val is None else validation_ema_beta * ema_val + (1.0 - validation_ema_beta) * best_epoch_f1
        print(f"Loss: {train_loss:.2f}, F1@0.5: {raw_f1:.2f}, Best F1: {best_epoch_f1:.3f}, Best thr: {epoch_threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
        scheduler.step(ema_val)
        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_threshold = epoch_threshold
            torch.save({"model": model.state_dict(), "vocab": vocab, "threshold": best_threshold, "best_f1": best_f1}, model_file_name)
    print(f"Training finished. Best validation F1: {best_f1:.3f}")

if __name__ == "__main__":
    main()


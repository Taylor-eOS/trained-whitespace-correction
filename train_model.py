import random
import struct
import string
import numpy as np
import torch
from tqdm import tqdm

val_file = "val.txt"
data_file = "train.bin"
index_file = "train.idx"
vocab_file = "vocab.pth"
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
pos_weight_value = 5.7
scheduler_patience = 4
scheduler_factor = 0.5
validation_ema_beta = 0.8
threshold_grid = np.linspace(0.05, 0.95, 19)
pad_id = None
unk_id = None

def load_index():
    with open(index_file, "rb") as f:
        num_lines = struct.unpack("Q", f.read(8))[0]
        vocab_size = struct.unpack("Q", f.read(8))[0]
        raw = f.read(num_lines * 10)
    entries = np.frombuffer(raw, dtype=np.uint8).reshape(num_lines, 10).copy()
    offsets = entries[:, :8].view(np.uint64).reshape(num_lines)
    lengths = entries[:, 8:10].view(np.uint16).reshape(num_lines)
    return offsets, lengths, vocab_size

class MMapFile:
    def __init__(self, path):
        import mmap
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def read_at(self, offset, nbytes):
        return self._mm[offset:offset + nbytes]

    def close(self):
        self._mm.close()
        self._f.close()

def _token_kind(token):
    if token == pad_token or token == "<pad>":
        return 0
    if token == "<unk>":
        return 6
    if not token:
        return 6
    ch = token[0]
    if ch.isspace():
        return 1
    if ch.isalpha():
        if ch.isupper():
            return 2
        return 3
    if ch.isdigit():
        return 4
    if ch in string.punctuation:
        return 5
    return 6

def build_kind_lookup(vocab):
    if isinstance(vocab, dict):
        vocab_size = max(vocab.values()) + 1
        id_to_token = [""] * vocab_size
        for token, idx in vocab.items():
            if 0 <= idx < vocab_size:
                id_to_token[idx] = str(token)
    else:
        id_to_token = [str(x) for x in vocab]
        vocab_size = len(id_to_token)
    kinds = torch.zeros(vocab_size, dtype=torch.long)
    for idx, token in enumerate(id_to_token):
        kinds[idx] = _token_kind(token)
    return kinds

def sample_batch_lines(data_fp, offsets, lengths, indices):
    inputs = []
    labels = []
    for idx in indices:
        length = int(lengths[idx])
        offset = int(offsets[idx])
        capped = min(length, max_length)
        tok_bytes = data_fp.read_at(offset, capped * 2)
        lab_bytes = data_fp.read_at(offset + length * 2, capped)
        tokens = np.frombuffer(tok_bytes, dtype=np.uint16).astype(np.int64)
        target = np.frombuffer(lab_bytes, dtype=np.uint8).astype(np.float32)
        if len(tokens) == 0:
            continue
        inputs.append(torch.from_numpy(tokens))
        labels.append(torch.from_numpy(target))
    if not inputs:
        return None, None
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=int(pad_id))
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)
    return input_padded, labels_padded

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
        self.blocks = torch.nn.ModuleList(
            [
                ConvBlock(channels, 3, d, dropout)
                for d in [1, 2, 4, 8, 16, 32]
            ]
        )
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

def estimate_pos_weight(data_fp, offsets, lengths, num_lines, prev=None, momentum=0.9, sample_size=2000):
    indices = random.sample(range(num_lines), min(sample_size, num_lines))
    total_pos = 0
    total_neg = 0
    for idx in indices:
        length = int(lengths[idx])
        offset = int(offsets[idx])
        capped = min(length, max_length)
        lab_bytes = data_fp.read_at(offset + length * 2, capped)
        labels = np.frombuffer(lab_bytes, dtype=np.uint8)
        total_pos += labels.sum()
        total_neg += len(labels) - labels.sum()
    if total_pos == 0:
        current = 1.0
    else:
        current = total_neg / total_pos
    if prev is None:
        return float(current)
    return float(momentum * prev + (1.0 - momentum) * current)

def train_epoch(model, optimizer, data_fp, offsets, lengths, num_lines):
    model.train()
    pos_weight_value = estimate_pos_weight(data_fp, offsets, lengths, num_lines)
    pos_weight = torch.tensor([pos_weight_value], device=next(model.parameters()).device)
    indices = random.sample(range(num_lines), min(lines_per_epoch, num_lines))
    progress_bar = tqdm(total=len(indices), desc=f"Training (pw={pos_weight_value:.2f})", unit="lines")
    total_loss = 0.0
    total_tokens = 0.0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        input_padded, labels_padded = sample_batch_lines(data_fp, offsets, lengths, batch_idx)
        if input_padded is None:
            continue
        logits = model(input_padded)
        mask = (input_padded != pad_id).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels_padded,
            pos_weight=pos_weight,
            reduction="none",
        )
        loss_sum = (loss * mask).sum()
        token_count = mask.sum()
        loss_mean = loss_sum / token_count
        optimizer.zero_grad(set_to_none=True)
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss_sum.item()
        total_tokens += token_count.item()
        progress_bar.update(len(batch_idx))
    progress_bar.close()
    return total_loss / total_tokens if total_tokens > 0 else 0.0

def prepare_val_batch(batch_lines, vocab):
    inputs = []
    labels = []
    for clean in batch_lines:
        if not clean:
            continue
        input_text = "".join(ch for ch in clean if ch != " ")
        if not input_text:
            continue
        input_text = input_text[:max_length]
        input_t = torch.tensor([vocab.get(c, unk_id) for c in input_text], dtype=torch.long)
        label_list = []
        kept = 0
        for i, ch in enumerate(clean):
            if ch == " ":
                continue
            nxt = clean[i + 1] if i + 1 < len(clean) else ""
            label_list.append(1.0 if nxt == " " else 0.0)
            kept += 1
            if kept >= len(input_text):
                break
        labels_t = torch.tensor(label_list, dtype=torch.float32)
        inputs.append(input_t)
        labels.append(labels_t)
    if not inputs:
        return None, None
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)
    return input_padded, labels_padded

def evaluate(model, vocab, val_lines):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.inference_mode():
        for start in range(0, len(val_lines), batch_size):
            batch_lines = val_lines[start:start + batch_size]
            input_padded, labels_padded = prepare_val_batch(batch_lines, vocab)
            if input_padded is None:
                continue
            logits = model(input_padded)
            mask = input_padded != pad_id
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
    best_f1 = -1.0
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
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
    return raw_f1, best_f1, best_threshold, raw_precision, raw_recall, predicted_positive_rate

def load_val_sample(val_size):
    reservoir = []
    with open(val_file, "r", encoding="latin-1") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if line == "":
                continue
            if len(reservoir) < val_size:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < val_size:
                    reservoir[j] = line
    return reservoir

def main():
    global pad_id, unk_id
    torch.set_num_threads(max(1, torch.get_num_threads()))
    checkpoint = torch.load(vocab_file, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    pad_id = vocab[pad_token]
    unk_id = vocab.get("<unk>", pad_id)
    offsets, lengths, vocab_size = load_index()
    num_lines = len(offsets)
    print(f"Loaded index: {num_lines:,} lines, vocab size: {vocab_size}")
    data_fp = MMapFile(data_file)
    val_lines = load_val_sample(val_size)
    kind_lookup = build_kind_lookup(vocab)
    model = WhitespaceCorrector(vocab_size, kind_lookup)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    best_f1 = 0.0
    ema_val = None
    best_threshold = 0.5
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Starting epoch {epoch + 1}/{num_epochs}  lr={current_lr:.2e}")
        train_loss = train_epoch(model, optimizer, data_fp, offsets, lengths, num_lines)
        print(f"Training loss: {train_loss:.6f}")
        raw_f1, best_epoch_f1, epoch_threshold, precision, recall, positive_rate = evaluate(model, vocab, val_lines)
        ema_val = best_epoch_f1 if ema_val is None else validation_ema_beta * ema_val + (1.0 - validation_ema_beta) * best_epoch_f1
        print(
            f"Epoch {epoch + 1} completed. "
            f"Validation F1@0.5: {raw_f1:.4f}  "
            f"Best F1: {best_epoch_f1:.4f}  "
            f"Best thr: {epoch_threshold:.2f}  "
            f"Precision: {precision:.4f}  "
            f"Recall: {recall:.4f}  "
            f"PosRate: {positive_rate:.4f}"
        )
        scheduler.step(ema_val)
        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_threshold = epoch_threshold
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "threshold": best_threshold,
                    "best_f1": best_f1,
                },
                "whitespace_corrector.pth",
            )
            print(f"  -> New best ({best_f1:.4f}), saved whitespace_corrector.pth")
    data_fp.close()
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "threshold": best_threshold,
            "best_f1": best_f1,
        },
        "whitespace_corrector.pth",
    )
    print(f"Training finished. Best validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()


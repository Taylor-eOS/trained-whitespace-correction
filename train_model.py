import torch
import random
import struct
import numpy as np
from tqdm import tqdm

val_file = "val.txt"
data_file = "train.bin"
index_file = "train.idx"
vocab_file = "vocab.pth"
val_size = 300
embedding_dim = 128
hidden_dim = 512
num_layers = 3
dropout = 0.3
learning_rate = 0.001
num_epochs = 40
batch_size = 32
max_length = 600
pad_token = "<pad>"
lines_per_epoch = 1000
pos_weight_value = 5.7

def load_index():
    with open(index_file, "rb") as f:
        num_lines = struct.unpack("Q", f.read(8))[0]
        vocab_size = struct.unpack("Q", f.read(8))[0]
        raw = f.read(num_lines * 10)
    entries = np.frombuffer(raw, dtype=np.uint8).reshape(num_lines, 10).copy()
    offsets = entries[:, :8].view(np.uint64).reshape(num_lines)
    lengths = entries[:, 8:10].view(np.uint16).reshape(num_lines)
    return offsets, lengths, vocab_size

def sample_batch_lines(data_fp, offsets, lengths, indices, max_length):
    inputs = []
    labelss = []
    for idx in indices:
        length = int(lengths[idx])
        offset = int(offsets[idx])
        capped = min(length, max_length)
        tok_bytes = data_fp.read_at(offset, capped * 2)
        lab_bytes = data_fp.read_at(offset + length * 2, capped)
        tokens = np.frombuffer(tok_bytes, dtype=np.uint16).astype(np.int64)
        labels = np.frombuffer(lab_bytes, dtype=np.uint8).astype(np.float32)
        inputs.append(torch.from_numpy(tokens))
        labelss.append(torch.from_numpy(labels))
    if not inputs:
        return None, None
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=int(pad_id))
    labels_padded = torch.nn.utils.rnn.pad_sequence(labelss, batch_first=True, padding_value=0.0)
    return input_padded, labels_padded

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

def compute_f1(logits, labels, mask):
    preds = (torch.sigmoid(logits) > 0.5).float()
    tp = ((preds == 1) & (labels == 1) & (mask == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0) & (mask == 1)).sum().item()
    fn = ((preds == 0) & (labels == 1) & (mask == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

class WhitespaceCorrector(torch.nn.Module):
    def __init__(self, char_vocab_size):
        super().__init__()
        channels = 256
        kernel_size = 3
        dilations = [1, 2, 4, 8, 16, 32]
        self.embedding = torch.nn.Embedding(char_vocab_size, embedding_dim)
        self.input_proj = torch.nn.Linear(embedding_dim, channels)
        self.blocks = torch.nn.ModuleList([
            self._make_block(channels, kernel_size, d) for d in dilations
        ])
        self.norm = torch.nn.LayerNorm(channels)
        self.head = torch.nn.Linear(channels, 1)

    def _make_block(self, channels, kernel_size, dilation):
        pad = dilation * (kernel_size - 1) // 2
        return torch.nn.Sequential(
            torch.nn.LayerNorm(channels),
            Transpose(),
            torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad, groups=channels),
            torch.nn.Conv1d(channels, channels, 1),
            Transpose(),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.input_proj(self.embedding(x))
        for block in self.blocks:
            out = out + block(out)
        return self.head(self.norm(out)).squeeze(-1)

class Transpose(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

def train_epoch(model, optimizer, data_fp, offsets, lengths, num_lines, pos_weight):
    model.train()
    indices = random.sample(range(num_lines), min(lines_per_epoch, num_lines))
    progress_bar = tqdm(total=len(indices), desc="Training", unit="lines")
    total_loss = 0.0
    total_tokens = 0.0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        input_padded, labels_padded = sample_batch_lines(data_fp, offsets, lengths, batch_idx, max_length)
        if input_padded is None:
            continue
        logits = model(input_padded)
        mask = (input_padded != pad_id).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels_padded, pos_weight=pos_weight, reduction="none"
        )
        loss_sum = (loss * mask).sum()
        token_count = mask.sum()
        loss_mean = loss_sum / token_count
        optimizer.zero_grad()
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
    labelss = []
    for clean in batch_lines:
        if not clean.strip():
            continue
        input_text = "".join(clean.split())
        input_len = len(input_text)
        if input_len > max_length:
            input_text = input_text[:max_length]
            input_len = max_length
        input_t = torch.tensor([vocab.get(c, pad_id) for c in input_text], dtype=torch.long)
        labels = []
        for k in range(len(clean)):
            if clean[k] != " ":
                label = 1 if k + 1 < len(clean) and clean[k + 1] == " " else 0
                labels.append(label)
                if len(labels) == input_len:
                    break
        labels_t = torch.tensor(labels, dtype=torch.float)
        inputs.append(input_t)
        labelss.append(labels_t)
    if not inputs:
        return None, None
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labelss, batch_first=True, padding_value=0.0)
    return input_padded, labels_padded

def validate(model, vocab, val_lines):
    model.eval()
    total_f1 = 0.0
    num_batches = 0
    sample = random.sample(val_lines, min(val_size, len(val_lines)))
    with torch.no_grad():
        for start in range(0, len(sample), batch_size):
            batch_lines = sample[start:start + batch_size]
            input_padded, labels_padded = prepare_val_batch(batch_lines, vocab)
            if input_padded is None:
                continue
            logits = model(input_padded)
            mask = (input_padded != pad_id).float()
            f1 = compute_f1(logits, labels_padded, mask)
            total_f1 += f1
            num_batches += 1
    return total_f1 / num_batches if num_batches > 0 else 0.0

def load_val_sample(val_size, vocab):
    reservoir = []
    with open(val_file, "r", encoding="latin-1") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if len(reservoir) < val_size:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < val_size:
                    reservoir[j] = line
    return reservoir

def main():
    checkpoint = torch.load(vocab_file, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    global pad_id
    pad_id = vocab[pad_token]
    offsets, lengths, vocab_size = load_index()
    num_lines = len(offsets)
    print(f"Loaded index: {num_lines:,} lines, vocab size: {vocab_size}")
    data_fp = MMapFile(data_file)
    val_lines = load_val_sample(val_size, vocab)
    model = WhitespaceCorrector(vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    pos_weight = torch.tensor([pos_weight_value])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    best_f1 = 0.0
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")
        train_loss = train_epoch(model, optimizer, data_fp, offsets, lengths, num_lines, pos_weight)
        print(f"Training loss: {train_loss:.6f}")
        avg_f1 = validate(model, vocab, val_lines)
        print(f"Epoch {epoch+1} completed. Validation F1: {avg_f1:.4f}")
        scheduler.step(avg_f1)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save({"model": model.state_dict(), "vocab": vocab}, "whitespace_corrector.pth")
            print(f"  -> New best ({best_f1:.4f}), saved whitespace_corrector.pth")
    data_fp.close()
    torch.save({"model": model.state_dict(), "vocab": vocab}, "whitespace_corrector.pth")
    print(f"Training finished. Best validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()


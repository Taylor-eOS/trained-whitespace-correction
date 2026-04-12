import torch
import random
import os
from tqdm import tqdm

train_file = "train.txt"
val_file = "val.txt"
val_size = 10000
embedding_dim = 64
hidden_dim = 128
num_layers = 2
learning_rate = 0.001
num_epochs = 5
batch_size = 16
max_length = 300
pad_token = "<pad>"
lines_per_epoch = 4000

def build_vocab(train_file):
    with open(train_file, "r", encoding="latin-1") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab = {char: idx for idx, char in enumerate(chars)}
    vocab[pad_token] = len(vocab)
    return vocab, len(vocab)

def prepare_batch(batch_lines, vocab, max_length):
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
        input_t = torch.tensor([vocab.get(c, vocab[pad_token]) for c in input_text], dtype=torch.long)
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
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=vocab[pad_token])
    labels_padded = torch.nn.utils.rnn.pad_sequence(labelss, batch_first=True, padding_value=0.0)
    return input_padded, labels_padded

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
        self.embedding = torch.nn.Embedding(char_vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        logits = self.linear(lstm_out)
        return logits.squeeze(-1)

def create_val_file(train_file, val_file, val_size):
    reservoir = []
    count = 0
    with open(train_file, "r", encoding="latin-1") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            count += 1
            if len(reservoir) < val_size:
                reservoir.append(stripped)
            else:
                j = random.randint(0, count - 1)
                if j < val_size:
                    reservoir[j] = stripped
    with open(val_file, "w", encoding="latin-1") as f:
        for line in reservoir:
            f.write(line + "\n")

def load_val_lines(val_file):
    lines = []
    with open(val_file, "r", encoding="latin-1") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines

def stream_train_lines(train_file, target_count):
    selected = []
    seen = set()
    total = 0
    with open(train_file, "r", encoding="latin-1") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            if len(selected) < target_count:
                selected.append(stripped)
                seen.add(stripped)
            else:
                j = random.randint(0, total - 1)
                if j < target_count:
                    replaced = selected[j]
                    selected[j] = stripped
                    seen.add(stripped)
    return selected, len(seen)

if not os.path.exists(val_file):
    print("Creating validation file...")
    create_val_file(train_file, val_file, val_size)

val_lines = load_val_lines(val_file)

vocab, vocab_size = build_vocab(train_file)
model = WhitespaceCorrector(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    train_lines, unique_count = stream_train_lines(train_file, lines_per_epoch)
    print(f"Unique lines this epoch: {unique_count}/{lines_per_epoch}")
    progress_bar = tqdm(total=len(train_lines), desc=f"Epoch {epoch+1} training", unit="lines")
    batch_lines = []
    for line in train_lines:
        batch_lines.append(line)
        if len(batch_lines) == batch_size:
            input_padded, labels_padded = prepare_batch(batch_lines, vocab, max_length)
            if input_padded is not None:
                logits = model(input_padded)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_padded, reduction="none")
                mask = (input_padded != vocab[pad_token]).float()
                loss = (loss * mask).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_lines = []
            progress_bar.update(batch_size)
    if batch_lines:
        input_padded, labels_padded = prepare_batch(batch_lines, vocab, max_length)
        if input_padded is not None:
            logits = model(input_padded)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_padded, reduction="none")
            mask = (input_padded != vocab[pad_token]).float()
            loss = (loss * mask).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        progress_bar.update(len(batch_lines))
    progress_bar.close()
    print("Training finished. Running validation...")
    model.eval()
    total_f1 = 0.0
    num_batches = 0
    with torch.no_grad():
        for start in range(0, len(val_lines), batch_size):
            batch_lines = val_lines[start:start + batch_size]
            input_padded, labels_padded = prepare_batch(batch_lines, vocab, max_length)
            if input_padded is None:
                continue
            logits = model(input_padded)
            mask = (input_padded != vocab[pad_token]).float()
            f1 = compute_f1(logits, labels_padded, mask)
            total_f1 += f1
            num_batches += 1
    avg_f1 = total_f1 / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {epoch+1} completed. Validation F1: {avg_f1:.4f}")

torch.save({"model": model.state_dict(), "vocab": vocab}, "whitespace_corrector.pth")
print("Training finished and model saved")

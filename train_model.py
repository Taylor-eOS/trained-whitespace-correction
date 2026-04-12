import torch
import random
from tqdm import tqdm
#from utils import 

val_file = "val.txt"
train_file = "train.txt"
val_size = 100
embedding_dim = 64
hidden_dim = 128
num_layers = 2
learning_rate = 0.001
num_epochs = 10
batch_size = 16
max_length = 300
pad_token = "<pad>"
lines_per_epoch = 1000

def build_vocab():
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

def stream_train_lines(target_count):
    selected = []
    with open(train_file, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        while len(selected) < target_count:
            pos = random.randint(0, file_size - 1)
            f.seek(pos)
            f.readline()
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                try:
                    stripped = line.decode("latin-1").strip()
                except Exception:
                    continue
                if not stripped:
                    continue
                selected.append(stripped)
                if len(selected) >= target_count:
                    break
    return selected

def train_epoch(model, optimizer, vocab):
    model.train()
    train_lines = stream_train_lines(lines_per_epoch)
    progress_bar = tqdm(total=len(train_lines), desc="Training", unit="lines")
    batch_lines = []
    total_loss = 0.0
    total_tokens = 0.0
    for line in train_lines:
        batch_lines.append(line)
        if len(batch_lines) == batch_size:
            input_padded, labels_padded = prepare_batch(batch_lines, vocab, max_length)
            if input_padded is not None:
                logits = model(input_padded)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_padded, reduction="none")
                mask = (input_padded != vocab[pad_token]).float()
                loss_sum = (loss * mask).sum()
                token_count = mask.sum()
                loss_mean = loss_sum / token_count
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()
                total_loss += loss_sum.item()
                total_tokens += token_count.item()
            batch_lines = []
            progress_bar.update(batch_size)
    if batch_lines:
        input_padded, labels_padded = prepare_batch(batch_lines, vocab, max_length)
        if input_padded is not None:
            logits = model(input_padded)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_padded, reduction="none")
            mask = (input_padded != vocab[pad_token]).float()
            loss_sum = (loss * mask).sum()
            token_count = mask.sum()
            loss_mean = loss_sum / token_count
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            total_loss += loss_sum.item()
            total_tokens += token_count.item()
        progress_bar.update(len(batch_lines))
    progress_bar.close()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    return avg_loss

def validate(model, vocab, val_lines):
    model.eval()
    total_f1 = 0.0
    num_batches = 0
    sample = random.sample(val_lines, min(val_size, len(val_lines)))
    with torch.no_grad():
        for start in range(0, len(sample), batch_size):
            batch_lines = sample[start:start + batch_size]
            input_padded, labels_padded = prepare_batch(batch_lines, vocab, max_length)
            if input_padded is None:
                continue
            logits = model(input_padded)
            mask = (input_padded != vocab[pad_token]).float()
            f1 = compute_f1(logits, labels_padded, mask)
            total_f1 += f1
            num_batches += 1
    return total_f1 / num_batches if num_batches > 0 else 0.0

def load_val_sample(val_size):
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
    with open(val_file, "r", encoding="latin-1") as f:
        val_lines = load_val_sample(val_size)
    vocab, vocab_size = build_vocab()
    model = WhitespaceCorrector(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, optimizer, vocab)
        print(f"Training loss: {train_loss:.6f}")
        print("Running validation...")
        avg_f1 = validate(model, vocab, val_lines)
        print(f"Epoch {epoch+1} completed. Validation F1: {avg_f1:.4f}")
    torch.save({"model": model.state_dict(), "vocab": vocab}, "whitespace_corrector.pth")
    print("Training finished and model saved")

if __name__ == "__main__":
    main()


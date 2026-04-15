import random
import numpy as np
import torch
from common import build_kind_lookup, load_text_lines
from common_inference import (WhitespaceCorrector, prepare_batch, run_batch_probs, pad_token, batch_size, max_length,)
from settings import train_file, val_file, vocab_file, model_file_name

val_size = 300
learning_rate = 3e-4
weight_decay = 1e-2
max_epochs = 500
target_f1 = 0.996
warmup_epochs = 4
lines_per_epoch = 1000
scheduler_patience = 4
scheduler_factor = 0.5
validation_ema_beta = 0.5
threshold_grid = np.linspace(0.05, 0.95, 19)
num_threads = 12
eval_noise_add_prob = 0.045
eval_noise_remove_prob = 0.02

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

def train_epoch(model, optimizer, train_lines, vocab, pad_id, unk_id, pos_weight):
    model.train()
    indices = random.sample(range(len(train_lines)), min(lines_per_epoch, len(train_lines)))
    total_loss = 0.0
    total_tokens = 0.0
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_lines = [train_lines[i] for i in batch_idx]
        input_padded, flags_padded, labels_padded, _ = prepare_batch(
            batch_lines, vocab, unk_id, pad_id, apply_noise=True
        )
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

def evaluate(model, vocab, pad_id, unk_id, val_lines):
    model.eval()
    sampled_lines = random.sample(val_lines, min(val_size, len(val_lines)))
    all_probs = []
    all_labels = []
    with torch.inference_mode():
        for start in range(0, len(sampled_lines), batch_size):
            batch_lines = sampled_lines[start:start + batch_size]
            input_padded, flags_padded, labels_padded, _ = prepare_batch(
                batch_lines, vocab, unk_id, pad_id, apply_noise=True,
                fixed_add_prob=eval_noise_add_prob, fixed_remove_prob=eval_noise_remove_prob,
            )
            if input_padded is None:
                continue
            per_row = run_batch_probs(model, input_padded, flags_padded, labels_padded)
            for row_probs, row_labels in per_row:
                if row_probs.size == 0:
                    continue
                all_probs.append(row_probs)
                all_labels.append(row_labels)
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
    model = WhitespaceCorrector(vocab_size, kind_lookup, head_bias_init, pad_id)
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
    for epoch in range(max_epochs):
        better_mark = ""
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = learning_rate * warmup_factor
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}, lr: {current_lr:.2e}")
        train_loss = train_epoch(model, optimizer, train_lines, vocab, pad_id, unk_id, pos_weight)
        raw_f1, best_epoch_f1, epoch_threshold, precision, recall, positive_rate = evaluate(model, vocab, pad_id, unk_id, val_lines)
        ema_val = best_epoch_f1 if ema_val is None else validation_ema_beta * ema_val + (1.0 - validation_ema_beta) * best_epoch_f1
        if epoch >= warmup_epochs:
            scheduler.step(ema_val)
        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_threshold = epoch_threshold
            torch.save({"model": model.state_dict(), "vocab": vocab, "threshold": best_threshold, "best_f1": best_f1}, model_file_name)
            better_mark = "*"
        print(f"Loss: {train_loss:.2f}, F1@0.5: {raw_f1:.2f}, best F1: {best_epoch_f1:.3f}{better_mark}, best thr: {epoch_threshold:.2f}, prec: {precision:.2f}, recall: {recall:.2f}")
        if best_epoch_f1 >= target_f1:
            print(f"Reached target F1 of {target_f1:.3f} at epoch {epoch + 1}.")
            break
    print(f"Training finished. Best validation F1: {best_f1:.3f}")

if __name__ == "__main__":
    main()

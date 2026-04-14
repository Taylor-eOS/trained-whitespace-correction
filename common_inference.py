import random
import numpy as np
import torch
from common import build_kind_lookup

embedding_dim = 128
kind_embedding_dim = 16
space_embedding_dim = 8
d_model = 256
nhead = 8
num_layers = 4
dim_feedforward = 768
head_dropout = 0.1
dropout = 0.1
max_length = 400
pad_token = "<pad>"
batch_size = 32

class WhitespaceCorrector(torch.nn.Module):
    def __init__(self, char_vocab_size, kind_lookup, head_bias_init, pad_id):
        super().__init__()
        self.pad_id = pad_id
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
        pad_mask = (x == self.pad_id)
        kind_ids = self.kind_lookup[x]
        char_emb = self.char_embedding(x)
        kind_emb = self.kind_embedding(kind_ids)
        space_emb = self.space_embedding(space_flags)
        pos_emb = self.pos_embedding(torch.arange(L, device=x.device).unsqueeze(0))
        out = self.input_proj(torch.cat([char_emb, kind_emb, space_emb], dim=-1)) + pos_emb
        out = self.transformer(out, src_key_padding_mask=pad_mask)
        out = self.head_drop(out)
        return self.head(out).squeeze(-1)

def prepare_batch(batch_lines, vocab, unk_id, pad_id, apply_noise, space_noise_add_prob_min=0.01, space_noise_add_prob_max=0.08, space_noise_remove_prob_min=0.00, space_noise_remove_prob_max=0.04, fixed_add_prob=None, fixed_remove_prob=None):
    inputs = []
    space_flags_list = []
    labels = []
    line_char_lists = []
    for line in batch_lines:
        chars = []
        flags = []
        char_list = []
        preceded_by_space = False
        for ch in line:
            if ch == " ":
                preceded_by_space = True
                continue
            chars.append(vocab.get(ch, unk_id))
            flags.append(1 if preceded_by_space else 0)
            char_list.append(ch)
            preceded_by_space = False
        targets = [0.0] * len(chars)
        for j in range(len(targets) - 1):
            if flags[j + 1] == 1:
                targets[j] = 1.0
        if apply_noise:
            add_p = fixed_add_prob if fixed_add_prob is not None else random.uniform(space_noise_add_prob_min, space_noise_add_prob_max)
            remove_p = fixed_remove_prob if fixed_remove_prob is not None else random.uniform(space_noise_remove_prob_min, space_noise_remove_prob_max)
            for j in range(len(flags)):
                if flags[j] == 0:
                    if random.random() < add_p:
                        flags[j] = 1
                else:
                    if random.random() < remove_p:
                        flags[j] = 0
        if flags and flags[0] == 1:
            flags[0] = 0
        chars = chars[:max_length]
        flags = flags[:max_length]
        targets = targets[:max_length]
        char_list = char_list[:max_length]
        if len(chars) == 0:
            continue
        inputs.append(torch.tensor(chars, dtype=torch.long))
        space_flags_list.append(torch.tensor(flags, dtype=torch.long))
        labels.append(torch.tensor(targets, dtype=torch.float32))
        line_char_lists.append(char_list)
    if not inputs:
        return None, None, None, []
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    flags_padded = torch.nn.utils.rnn.pad_sequence(space_flags_list, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0.0)
    return input_padded, flags_padded, labels_padded, line_char_lists

def run_batch_probs(model, input_padded, flags_padded, labels_padded):
    pad_id = model.pad_id
    logits = model(input_padded, flags_padded)
    mask = (input_padded != pad_id)
    probs = torch.sigmoid(logits)
    result = []
    for i in range(input_padded.shape[0]):
        row_mask = mask[i]
        row_probs = probs[i][row_mask].detach().cpu().numpy()
        row_labels = labels_padded[i][row_mask].detach().cpu().numpy().astype(np.int32)
        result.append((row_probs, row_labels))
    return result

def load_model(model_file, device):
    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    threshold = checkpoint["threshold"]
    pad_id = vocab[pad_token]
    unk_id = vocab.get("<unk>", pad_id)
    vocab_size = max(vocab.values()) + 1 if isinstance(vocab, dict) else len(vocab)
    kind_lookup = build_kind_lookup(vocab, pad_token)
    model = WhitespaceCorrector(vocab_size, kind_lookup, head_bias_init=0.0, pad_id=pad_id)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, vocab, unk_id, threshold

def correct_lines(model, lines, vocab, unk_id, threshold):
    corrected = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(lines), batch_size):
            batch_lines = lines[start:start + batch_size]
            input_padded, flags_padded, labels_padded, char_lists = prepare_batch(batch_lines, vocab, unk_id, model.pad_id, apply_noise=False)
            if input_padded is None:
                corrected.extend([""] * len(batch_lines))
                continue
            input_padded = input_padded.to(device)
            flags_padded = flags_padded.to(device)
            labels_padded = labels_padded.to(device)
            per_row = run_batch_probs(model, input_padded, flags_padded, labels_padded)
            for (row_probs, _), char_list in zip(per_row, char_lists):
                space_after = row_probs >= threshold
                out = []
                for j, ch in enumerate(char_list):
                    out.append(ch)
                    if j < len(space_after) and space_after[j]:
                        out.append(" ")
                corrected.append("".join(out))
    return corrected

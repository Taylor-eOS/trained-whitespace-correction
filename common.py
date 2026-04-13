import torch
import string

def _token_kind(token, pad_token):
    if token == pad_token or token == "<pad>": return 0
    if token == "<unk>": return 6
    if not token: return 6
    ch = token[0]
    if ch.isspace(): return 1
    if ch.isalpha():
        if ch.isupper(): return 2
        return 3
    if ch.isdigit(): return 4
    if ch in string.punctuation: return 5
    return 6

def build_kind_lookup(vocab, pad_token):
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
        kinds[idx] = _token_kind(token, pad_token)
    return kinds

def load_text_lines(filepath):
    with open(filepath, "r", encoding="latin-1") as f:
        return [line.rstrip("\n") for line in f if line.strip()]

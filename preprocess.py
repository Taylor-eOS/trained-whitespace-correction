import struct
import numpy as np

train_file = "train.txt"
out_data = "train.bin"
out_index = "train.idx"
pad_token = "<pad>"

def build_vocab(raw):
    present = np.zeros(256, dtype=bool)
    present[raw] = True
    byte_vals = sorted(int(b) for b in np.where(present)[0])
    vocab = {chr(b): i for i, b in enumerate(byte_vals)}
    vocab[pad_token] = len(vocab)
    return vocab

def preprocess():
    print(f"Reading {train_file} ...")
    with open(train_file, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    print(f"File size: {len(raw):,} bytes")
    print("Building vocab ...")
    vocab = build_vocab(raw)
    pad_id = vocab[pad_token]
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
    byte_to_id = np.full(256, pad_id, dtype=np.uint16)
    for ch, idx in vocab.items():
        if ch != pad_token:
            byte_to_id[ord(ch)] = idx
    space = np.uint8(0x20)
    newline = np.uint8(0x0A)
    print("Locating line boundaries ...")
    newline_positions = np.where(raw == newline)[0]
    line_starts = np.empty(len(newline_positions) + 1, dtype=np.int64)
    line_starts[0] = 0
    line_starts[1:] = newline_positions + 1
    line_ends = np.empty(len(newline_positions) + 1, dtype=np.int64)
    line_ends[:-1] = newline_positions
    line_ends[-1] = len(raw)
    print(f"Total lines (including empty): {len(line_starts):,}")
    index_entries = []
    data_offset = 0
    skipped = 0
    print("Tokenizing ...")
    with open(out_data, "wb") as fdata:
        for i in range(len(line_starts)):
            start = int(line_starts[i])
            end = int(line_ends[i])
            if end > start and raw[end - 1] == 0x0D:
                end -= 1
            line = raw[start:end]
            if len(line) == 0:
                skipped += 1
                continue
            non_space_mask = line != space
            if not non_space_mask.any():
                skipped += 1
                continue
            tokens = byte_to_id[line[non_space_mask]]
            length = len(tokens)
            shifted = np.empty(len(line), dtype=np.uint8)
            shifted[:-1] = line[1:]
            shifted[-1] = 0
            is_space_after = (shifted == space).astype(np.uint8)
            labels = is_space_after[non_space_mask]
            tok_bytes = tokens.tobytes()
            lab_bytes = labels.tobytes()
            fdata.write(tok_bytes)
            fdata.write(lab_bytes)
            index_entries.append((data_offset, length))
            data_offset += len(tok_bytes) + len(lab_bytes)
            if (i + 1) % 1_000_000 == 0:
                print(f"  {i+1:,} lines processed, {len(index_entries):,} kept ...")
    print(f"Writing index ({len(index_entries):,} lines, {skipped:,} skipped) ...")
    with open(out_index, "wb") as fidx:
        fidx.write(struct.pack("Q", len(index_entries)))
        fidx.write(struct.pack("Q", vocab_size))
        for offset, length in index_entries:
            fidx.write(struct.pack("QH", offset, length))
    import torch
    torch.save({"vocab": vocab}, "vocab.pth")
    print(f"Done. Wrote {out_data}, {out_index}, vocab.pth")
    print(f"  Lines: {len(index_entries):,}, vocab size: {vocab_size}")

if __name__ == "__main__":
    preprocess()


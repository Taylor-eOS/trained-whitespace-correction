import sys
import random
from pathlib import Path
import codecs

DEFAULT_CHUNK_SIZE = 64 * 1024
corpus_file = "corpus.txt"
VAL_FRACTION = 0.01

def convert_to_latin1(input_path):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    result = []
    with open(input_path, "rb") as inf:
        while True:
            raw_chunk = inf.read(DEFAULT_CHUNK_SIZE)
            if not raw_chunk:
                break
            text_chunk = decoder.decode(raw_chunk, final=False)
            latin1_chunk = text_chunk.encode("latin-1", errors="ignore")
            result.append(latin1_chunk)
        final_text = decoder.decode(b"", final=True)
        if final_text:
            result.append(final_text.encode("latin-1", errors="ignore"))
    return b"".join(result)

def split_val(full_text, val_fraction):
    lines = [line.strip() for line in full_text.decode("latin-1").splitlines() if line.strip()]
    if not lines:
        return b"", b""
    val_size = max(1, int(len(lines) * val_fraction))
    reservoir = []
    for i, line in enumerate(lines):
        if len(reservoir) < val_size:
            reservoir.append(line)
        else:
            j = random.randint(0, i)
            if j < val_size:
                reservoir[j] = line
    val_set = set(reservoir)
    val_bytes = "\n".join(reservoir).encode("latin-1") + b"\n"
    train_lines = [l for l in lines if l not in val_set]
    train_bytes = "\n".join(train_lines).encode("latin-1") + b"\n"
    return train_bytes, val_bytes

def main():
    input_str = input(f"Input file ({corpus_file}): ") or corpus_file
    input_file = Path(input_str)
    if not input_file.is_file():
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    print(f"Converting '{input_file}' to Latin-1...")
    full_latin1 = convert_to_latin1(input_file)
    print("Splitting into train and val sets...")
    train_bytes, val_bytes = split_val(full_latin1, VAL_FRACTION)
    with open("train.txt", "wb") as f:
        f.write(train_bytes)
    with open("val.txt", "wb") as f:
        f.write(val_bytes)
    print(f"Val set: {len(val_bytes.splitlines())} lines. Train set: {len(train_bytes.splitlines())} lines.")
    print("Output written to 'train.txt' and 'val.txt'.")

if __name__ == "__main__":
    main()


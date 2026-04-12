import sys
import random
from pathlib import Path
import codecs

DEFAULT_CHUNK_SIZE = 64 * 1024
corpus_file = "corpus.txt"
VAL_FRACTION = 0.01

def convert_to_latin1(input_path, output_path, chunk_size=DEFAULT_CHUNK_SIZE):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with open(input_path, "rb") as inf, open(output_path, "wb") as outf:
        while True:
            raw_chunk = inf.read(chunk_size)
            if not raw_chunk:
                break
            text_chunk = decoder.decode(raw_chunk, final=False)
            latin1_chunk = text_chunk.encode("latin-1", errors="ignore")
            outf.write(latin1_chunk)
        final_text = decoder.decode(b"", final=True)
        if final_text:
            outf.write(final_text.encode("latin-1", errors="ignore"))

def split_val(train_path, val_path, val_fraction):
    all_lines = []
    with open(train_path, "r", encoding="latin-1") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                all_lines.append(stripped)
    val_size = max(1, int(len(all_lines) * val_fraction))
    reservoir = []
    for i, line in enumerate(all_lines):
        if len(reservoir) < val_size:
            reservoir.append(line)
        else:
            j = random.randint(0, i)
            if j < val_size:
                reservoir[j] = line
    val_set = set(reservoir)
    with open(val_path, "w", encoding="latin-1") as f:
        for line in reservoir:
            f.write(line + "\n")
    train_lines = [l + "\n" for l in all_lines if l not in val_set]
    with open(train_path, "w", encoding="latin-1") as f:
        f.writelines(train_lines)
    print(f"Val set: {len(reservoir)} lines. Train set: {len(train_lines)} lines.")

def main():
    input_str = input(f"Input file ({corpus_file}): ") or corpus_file
    input_file = Path(input_str)
    if not input_file.is_file():
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    print(f"Converting '{input_file}' to Latin-1...")
    convert_to_latin1(input_file, "train.txt")
    print("Splitting val set...")
    split_val("train.txt", "val.txt", VAL_FRACTION)
    print("Output written to 'train.txt' and 'val.txt'.")

if __name__ == "__main__":
    main()


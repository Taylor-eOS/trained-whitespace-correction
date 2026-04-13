import sys
import random
from pathlib import Path
import codecs
import unicodedata
from collections import Counter

DEFAULT_CHUNK_SIZE = 64 * 1024
corpus_file = "corpus.txt"
VAL_FRACTION = 0.01
DEFAULT_CHUNK_SIZE = 1 << 20
ALLOWED_CHARS = [" ", "\n", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "æ", "ø", "å", "Æ", "Ø", "Å", "ä", "ö", "ü", "Ä", "Ö", "Ü", "ß", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", ":", ";", "!", "?", "'", "\"", "-", "_", "(", ")", "/", "&", "*", "$", "£", "[", "]", "<", ">", "{", "}", "%", "~", "#", "+", "=", "@", "\\", "^", "|", "°", "±", "©", "®", "¶", "¬", "¦", "¢", "¥", "¤", "¡", "¿"]
DIRECT_REPLACEMENTS = {"\r\n": "\n", "\r": "\n", "\t": " ", "\u00A0": " ", "\u2000": " ", "\u2001": " ", "\u2002": " ", "\u2003": " ", "\u2004": " ", "\u2005": " ", "\u2006": " ", "\u2007": " ", "\u2008": " ", "\u2009": " ", "\u200A": " ", "\u202F": " ", "\u205F": " ", "\u3000": " ", "\u200B": "", "\u200C": "", "\u200D": "", "\uFEFF": "", "\u00AD": "", "\u2028": "\n", "\u2029": "\n", "\u0060": "'", "\u00B4": "'", "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'", "\u2039": "'", "\u203A": "'", "\u201C": "\"", "\u201D": "\"", "\u201E": "\"", "\u201F": "\"", "\u00AB": "\"", "\u00BB": "\"", "\u2014": "-", "\u2013": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2015": "-", "\u2026": "...", "\u00B7": ".", "\u2022": "*", "\u00A3": "£", "\u00A7": "$", "\u00D7": "*", "\u2044": "/", "\u00F0": "d", "\u00D0": "D", "\u00FE": "th", "\u00DE": "Th", "\u00F7": "/", "\u03BC": "u", "\u0153": "oe", "\u0152": "OE", "\u0142": "l", "\u0141": "L", "\u0111": "d", "\u0110": "D", "\u0127": "h", "\u0126": "H", "\u0167": "t", "\u0166": "T", "\u0180": "b", "\u0243": "B", "\u0268": "i", "\u0197": "I", "\u0289": "u", "\u0244": "U", "\u0180": "b", "\u0182": "B", "\u023c": "c", "\u0188": "c", "\u0187": "C", "\u0256": "d", "\u0257": "d", "\u0493": "g", "\u0492": "G", "\u0266": "h", "\u0267": "h", "\u029d": "j", "\u0199": "k", "\u0198": "K", "\u026b": "l", "\u019a": "l", "\u0140": "l", "\u013f": "L", "\u0271": "m", "\u019d": "N", "\u0273": "n", "\u0275": "o", "\u01eb": "o", "\u2c63": "P", "\u1d7d": "p", "\u024d": "r", "\u024c": "R", "\u023f": "s", "\u0240": "z", "\u0288": "t", "\u0163": "t", "\u0162": "T", "\u0289": "u", "\u028b": "v", "\u1d8c": "v", "\u0263": "g", "\u0260": "g", "\u0262": "G", "\u00A8": "\"", "\u00AA": "a", "\u00AF": "-", "\u00B2": "2", "\u00B3": "3", "\u00B5": "u", "\u00B8": ",", "\u00B9": "1", "\u00BA": "o", "\u00BC": "1/4", "\u00BD": "1/2", "\u00BE": "3/4"}
ALLOWED_SET = set(ALLOWED_CHARS)
PRECOMPOSED = set("æøåÆØÅäöüÄÖÜß")

def _normalize_text(text):
    for src, dst in DIRECT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    result = []
    for c in text:
        if c in ALLOWED_SET or c in PRECOMPOSED:
            result.append(c)
            continue
        decomposed = unicodedata.normalize("NFD", c)
        base = decomposed[0]
        if base in ALLOWED_SET and all(unicodedata.combining(d) for d in decomposed[1:]):
            result.append(base)
    return "".join(result)

def _collapse_whitespace(text):
    lines = []
    for raw_line in text.split("\n"):
        parts = raw_line.split()
        lines.append(" ".join(parts))
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"

def convert_to_latin1(input_path):
    with open(input_path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    text = _normalize_text(text)
    text = _collapse_whitespace(text)
    return text.encode("latin-1", errors="ignore")

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


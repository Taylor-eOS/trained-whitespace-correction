import sys
import os
import torch
from common import load_text_lines
from common_inference import load_model, correct_lines
from settings import model_file_name

num_threads = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def correct_file(input_path, output_path, model, vocab, unk_id, threshold):
    lines = load_text_lines(input_path)
    print(f"Loaded {len(lines)} lines from {input_path}")
    corrected = correct_lines(model, lines, vocab, unk_id, threshold)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in corrected:
            f.write(line + "\n")
    print(f"Written corrected output to {output_path}")

def main():
    torch.set_num_threads(num_threads)
    input_path = input("Input file (input.txt): ") or "input.txt"
    base, ext = os.path.splitext(input_path)
    output_path = base + "_corrected" + ext
    model, vocab, unk_id, threshold = load_model(model_file_name, device)
    print(f"Loaded model (threshold={threshold:.3f})")
    correct_file(input_path, output_path, model, vocab, unk_id, threshold)

if __name__ == "__main__":
    main()

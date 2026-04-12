import sys

def main():
    input_file = "files/corpus.txt"
    output_file = "train.txt"
    target_mb = int(input("Target MB: "))
    target_bytes = target_mb * 1024 * 1024
    written = 0
    chunk_size = 1024 * 1024
    with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
        while written < target_bytes:
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            to_write = min(len(chunk), target_bytes - written)
            outfile.write(chunk[:to_write])
            written += to_write
            if to_write < len(chunk):
                break
    print(f"Created {output_file} with approximately {written / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()

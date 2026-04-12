import random

def create_val_set(train_file, val_size):
    reservoir = []
    count = 0
    with open(train_file, "r", encoding="latin-1") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            count += 1
            if len(reservoir) < val_size:
                reservoir.append(stripped)
            else:
                j = random.randint(0, count - 1)
                if j < val_size:
                    reservoir[j] = stripped
    return set(reservoir)

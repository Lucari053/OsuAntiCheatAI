import json
import random

def split_replays(labels_path: str, val_ratio: float = 0.2, seed: int = 42):
    """
    Split replay into train/val split
    Args:
        labels_path (str):   label json path
        val_ratio   (float): split value, if greater than 1, act like val counter
    """
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    cheaters = [h for h, v in labels.items() if v]
    legits   = [h for h, v in labels.items() if not v]

    rng = random.Random(seed)
    rng.shuffle(cheaters)
    rng.shuffle(legits)

    def cut(lst):
        if not lst: return [], []
        if val_ratio > 1:
            n_val = int(val_ratio)
        else:
            n_val = max(1, int(len(lst) * val_ratio))
        return lst[n_val:], lst[:n_val]
    
    train_c, val_c = cut(cheaters)
    train_l, val_l = cut(legits)

    return set(train_c + train_l), set(val_c + val_l) 
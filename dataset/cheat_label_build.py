import json
import os
import argparse

from dataset.utils import ZarrReplay


def build_label(legit_zarr: str, cheat_zarr: str, json_out: str, limit: int = None):
    """
    The limit is per label *cheat*/*legit*
    """

    label = {}
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    
    legit_dataset = ZarrReplay(legit_zarr)
    cheat_dataset = ZarrReplay(cheat_zarr)

    for r_hash in cheat_dataset.z_hash:
        label[r_hash] = True
        if limit and len(label) >= limit:
            break
    
    cheat_count = len(label)
    for r_hash in legit_dataset.z_hash:
        label[r_hash] = False
        if limit and len(label) - cheat_count >= limit:
            break
    
    # Not exaclty equal because of MD5 hashing
    print(f"Total: {len(label)}, cheat: {cheat_count}, legit: {len(label)-cheat_count}") 
    
    with open(json_out, 'w') as f:
        json.dump(label, f, indent=2)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-legit-zarr", required=True, type=str)
    parser.add_argument("--replay-cheat-zarr", required=True, type=str)
    parser.add_argument("--out-label-json",    required=True, type=str, help="The path of the output label file (.json)")
    parser.add_argument("--limit-per-label",   required=False, type=int, help="The limit per label, for exemple 100k become 100k cheat, 100k legit.")
    
    args = parser.parse_args()

    build_label(
        args.replay_legit_zarr,
        args.replay_cheat_zarr,
        args.out_label_json,
        args.limit_per_label
    )
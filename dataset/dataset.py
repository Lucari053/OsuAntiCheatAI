import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.utils import ZarrBeatmap, ZarrReplay, ZarrPair

class ReplaySequenceDataset(Dataset):
    def __init__(
        self, 
        pairs_zarr:    str, 
        legit_zarr:    str,
        cheat_zarr:    str, 
        beatmaps_zarr: str, 
        labels_path:     str = None,
        allowed_replays: set = None,
        max_windows:     int = None
    ):
        print(f"-- Load Pairs dataset --")
        
        self.pair_dataset    = ZarrPair(pairs_zarr)
        self.legit_dataset   = ZarrReplay(legit_zarr)
        self.cheat_dataset   = ZarrReplay(cheat_zarr)
        self.beatmap_dataset = ZarrBeatmap(beatmaps_zarr)
        self.max_windows     = max_windows

        # Build index maps for all datasets
        self.pair_dataset   .build_hash_to_idx()
        self.legit_dataset  .build_hash_to_idx()
        self.cheat_dataset  .build_hash_to_idx()
        self.beatmap_dataset.build_hash_to_idx()

        # Load all pair hashes and their beatmap hashes in bulk to avoid slow disk reads
        pair_hashes = self.pair_dataset.z_hash[:]
        pair_beatmaps = self.pair_dataset.z_beatmap[:]
        pair_hash_to_beatmap = dict(zip(pair_hashes, pair_beatmaps))

        # Get sets of existing hashes for O(1) in-memory lookups
        legit_hashes = set(self.legit_dataset.hash_to_idx.keys())
        cheat_hashes = set(self.cheat_dataset.hash_to_idx.keys())
        beatmap_hashes = set(self.beatmap_dataset.hash_to_idx.keys())

        with open(labels_path, 'r') as f:
            labels_dict = json.load(f)
        
        self.labels = []
        for r_hash, bcheat in labels_dict.items():
            # Filter by allowed_replays (keep only if in the set)
            if allowed_replays is not None and r_hash not in allowed_replays:
                continue

            # In-memory fast checks
            b_hash = pair_hash_to_beatmap.get(r_hash)
            if b_hash is not None and b_hash in beatmap_hashes:
                r_set = cheat_hashes if bcheat else legit_hashes
                if r_hash in r_set:
                    self.labels.append((r_hash, bcheat))

        self.n_cheat = sum([bcheat for _, bcheat in self.labels])

    def class_counts(self):
        return len(self.labels) - self.n_cheat, self.n_cheat

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        r_hash, bcheat = self.labels[idx]
        
        pair_info = self.pair_dataset[r_hash]
        pairs_arr = pair_info['pairs']
        
        bd_arr = self.beatmap_dataset[pair_info['beatmap']]['objects']
        
        if bcheat:
            rd_arr = self.cheat_dataset[pair_info['hash']]['datas']
        else:
            rd_arr = self.legit_dataset[pair_info['hash']]['datas']
        
        if self.max_windows is not None and len(pairs_arr) > self.max_windows:
            idxs = np.linspace(0, len(pairs_arr) - 1, self.max_windows, dtype=int)
            pairs_arr = pairs_arr[idxs]
        
        beatmap_windows = np.stack([bd_arr[b0:b0 + bc] for b0, bc, _, _ in pairs_arr])
        replay_windows  = np.stack([rd_arr[r0:r0 + rc] for _, _, r0, rc in pairs_arr])

        # TODO: correct error directly on dataset generation
        # Normalization and clipping to prevent numerical instabilities
        beatmap_windows[:, :, 2] = np.clip(beatmap_windows[:, :, 2], 0.0, 50.0) / 50.0
        
        replay_windows[:, :, 2] = np.clip(replay_windows[:, :, 2], 0.0, 50.0) / 50.0
        replay_windows[:, :, 10] = np.clip(replay_windows[:, :, 10], 0.0, 10.0) / 10.0
        replay_windows[:, :, 11] = np.clip(replay_windows[:, :, 11], -10.0, 10.0) / 10.0
        replay_windows[:, :, 12] = np.clip(replay_windows[:, :, 12], -10.0, 10.0) / 10.0

        return (
            torch.tensor(beatmap_windows, dtype=torch.float32),
            torch.tensor(replay_windows, dtype=torch.float32),
            torch.tensor(bcheat, dtype=torch.float32)
        )
    
def collate_replays(batch):
    beatmap_list, replay_list, labels = zip(*batch)
    lengths = [b.shape[0] for b in beatmap_list]
    max_n = max(lengths)
    B = len(batch)
    n_object, n_feat_b = beatmap_list[0].shape[1:]
    n_frame, n_feat_r  = replay_list [0].shape[1:]

    beatmap_batch = torch.zeros(B, max_n, n_object, n_feat_b)
    replay_batch  = torch.zeros(B, max_n, n_frame,  n_feat_r)
    mask = torch.ones(B, max_n, dtype=torch.bool)

    for i, (b, r, n) in enumerate(zip(beatmap_list, replay_list, lengths)):
        beatmap_batch[i, :n] = b
        replay_batch [i, :n] = r
        mask         [i, :n] = False
    
    return beatmap_batch, replay_batch, mask, torch.stack(labels)


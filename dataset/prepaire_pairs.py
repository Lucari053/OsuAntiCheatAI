import numpy as np
import os
import json
from pebble import ProcessPool
from concurrent.futures import TimeoutError, as_completed
from tqdm import tqdm
import argparse

from config import Config
from dataset.utils import ReplayFrame, BeatmapObject, Pairs, ZarrReplay, ZarrBeatmap, ZarrPair

# -- Workers-global state (initialized once per process) --
_worker_beatmap = None
_worker_legit   = None
_worker_cheat   = None

def _init_worker(replay_legit_path: str, replay_cheat_path: str, beatmap_path: str):
    """Open ZarrBeatmap once per worker process instead of per task."""
    global _worker_legit, _worker_cheat, _worker_beatmap
    _worker_legit = ZarrReplay(replay_legit_path)
    _worker_cheat = ZarrReplay(replay_cheat_path)
    _worker_beatmap = ZarrBeatmap(beatmap_path)
    _worker_beatmap.build_hash_to_idx()

def prepair_pairs(replay_data: np.ndarray, beatmap_objects: np.ndarray) -> np.ndarray:
    """
    Create replay/beatmap pairs

    Returns:
        pairs (np.ndarray): [(beatmap_start, beatmap_count, replay_start, replay_count),...]
    """
    frame_ctx  = Config.get("data/n_frame")  // 2
    object_ctx = Config.get("data/n_object") // 2

    if len(beatmap_objects) < 2 * object_ctx or len(replay_data) == 0:
        return None
    
    # beatmap objects cumsum (t -> index 2)
    bd_t = beatmap_objects[:, 2]
    bd_cum = np.cumsum(bd_t)

    # replay cumsum (t -> index 2)
    rd_t = replay_data[:, 2]
    rd_cum = np.cumsum(rd_t)
      
    # obj_time values
    obj_times = bd_cum[object_ctx : len(beatmap_objects) - object_ctx]
    i_values = np.arange(object_ctx, len(beatmap_objects) - object_ctx, dtype=np.int32)

    # Find where each obj_time fits in rd_cum
    l_r_idx = np.searchsorted(rd_cum, obj_times, side='right') + 1
    l_r_idx = np.minimum(len(replay_data), l_r_idx)

    
    break_cond = (l_r_idx + frame_ctx >= len(replay_data))
    break_indices = np.where(break_cond)[0]
    if len(break_indices) > 0:
        cutoff = break_indices[0]
        l_r_idx = l_r_idx[:cutoff]
        i_values = i_values[:cutoff]

    valid_mask = (l_r_idx - frame_ctx >= 0)
    l_r_idx = l_r_idx[valid_mask]
    i_values = i_values[valid_mask]

    if len(l_r_idx) == 0:
        return None
    
    # Construct the pairs matrix: (beatmap_start, beatmap_count, replay_start, replay_count)
    col0 = i_values - object_ctx
    col1 = np.full_like(col0, object_ctx * 2)
    col2 = l_r_idx - frame_ctx
    col3 = np.full_like(col0, frame_ctx * 2)

    pairs = np.column_stack([col0, col1, col2, col3]).astype(np.int32)
    return pairs


def prepair_pairs_zarr(bcheat_dataset: bool, replay_idx: int):

    # Load data from replay
    r_dataset = _worker_cheat if bcheat_dataset else _worker_legit
    rd = r_dataset[replay_idx]

    rd_datas = rd['datas']
    r_hash  = rd['hash']
    b_hash  = rd['beatmap']

    # Load beatmap objects
    bd_data = _worker_beatmap[b_hash]
    bd_objects = bd_data['objects']
    
    pairs = prepair_pairs(rd_datas, bd_objects)
    
    return r_hash, pairs, b_hash




class PrepairPairs:

    def process_json(
        self,
        json_path:      str,
        legit_zarr:     str,
        cheat_zarr:     str,
        beatmap_zarr:    str,
        out_pairs_zarr: str,
        max_workers:    int = None,
        timeout:        int = 2,
    ):
        """
        Process pairs with the label json file.
        """

        os.makedirs(os.path.dirname(out_pairs_zarr), exist_ok=True)

        legit_dataset = ZarrReplay(legit_zarr)
        cheat_dataset = ZarrReplay(cheat_zarr)
        pair_dataset  = ZarrPair(out_pairs_zarr)

        legit_dataset.build_hash_to_idx()
        cheat_dataset.build_hash_to_idx()

        with open(json_path, 'r') as f:
            r_hashs = json.load(f)
        
        args = []
        for r_hash, bcheat in r_hashs.items():
            
            if r_hash in pair_dataset:
                continue
            
            idx = None
            if bcheat:
                idx = cheat_dataset.hash_to_idx.get(r_hash)
            
            else:
                idx = legit_dataset.hash_to_idx.get(r_hash)

            if idx is None: continue

            # Args: (replay_zarr, replay_idx, beatmap_zarr)
            args.append((bcheat, idx))


        with ProcessPool(
            max_workers=max_workers or os.cpu_count(),
            initializer=_init_worker,
            initargs=(legit_zarr, cheat_zarr, beatmap_zarr)
        ) as pool:
            futures = {pool.schedule(prepair_pairs_zarr, args=a, timeout=timeout) for a in args}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = fut.result()
                    if result is not None:
                        pair_dataset += result

                except TimeoutError:
                    pass
                except Exception:
                    pass
            
            pair_dataset.flush_batch()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--label-json",        required=True, type=str)
    parser.add_argument("--replay-legit-zarr", required=True, type=str)
    parser.add_argument("--replay-cheat-zarr", required=True, type=str)
    parser.add_argument("--beatmap-zarr",      required=True, type=str)
    parser.add_argument("--out-pairs-zarr",    required=True, type=str)
    parser.add_argument("--num-workers",       required=False, type=int, default=None)

    args = parser.parse_args()

    prep = PrepairPairs()
    prep.process_json(
        args.label_json,
        args.replay_legit_zarr,
        args.replay_cheat_zarr,
        args.beatmap_zarr,
        args.out_pairs_zarr,
        args.num_workers
    )
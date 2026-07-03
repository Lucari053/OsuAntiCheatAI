import os
import hashlib
from pebble import ProcessPool
from concurrent.futures import TimeoutError, as_completed
from tqdm import tqdm
import numpy as np
from osrparse import Replay, GameMode
import pickle
import argparse

from dataset.prepaire_beatmap import download_beatmap
from dataset.utils import BeatmapObject, ZarrReplay, ZarrBeatmap
from dataset.mods import is_supported, apply_mods_to_difficulty
from parser.edit_osr import get_beatmap_replay_hash


# -- Worker-global state (initialized once per process) --
_worker_beatmap = None

def _init_worker(beatmap_zarr_path: str):
    """Open ZarrBeatmap once per worker process instead of per task."""
    global _worker_beatmap
    _worker_beatmap = ZarrBeatmap(beatmap_zarr_path)


def validate_replay(
    replay:  Replay,
    beatmap: list[BeatmapObject],
    max_miss_ratio:   float = 0.3,
    max_avg_frame_ms: float = 30.0,
    min_duration_ms:  float = 15_000,
    max_coord:        float = 1000.0
) -> tuple[bool, str]:
    
    if replay.mode != GameMode.STD:
        return False, "wrong_mode"
    if not is_supported(replay.mods):
        return False, "unsupported_mods"
    
    total_objects = (
        sum(1 for o in beatmap if o.object_type in (0, 1))
        + sum(1 for o in beatmap if o.object_type == 4) // 2
    )
    if total_objects == 0:
        return False, "empty_beatmap"
    
    # Check if player clear the map
    total_judged = replay.count_300 + replay.count_100 + replay.count_50 + replay.count_miss
    if total_judged < total_objects:
        return False, "incomplete_play"
    
    if replay.count_miss / total_objects > max_miss_ratio:
        return False, "too_many_misses"
    
    frames = [f for f in replay.replay_data if f.time_delta != -12345]
    if len(frames) < 2:
        return False, "too_few_frames"
    
    total_time = sum(f.time_delta for f in frames)
    if total_time < min_duration_ms:
        return False, "too_short"
    
    avg_frame_ms = total_time / len(frames)
    if avg_frame_ms > max_avg_frame_ms:
        return False, "low_frame_rate"
    
    for f in frames:
        if abs(f.x) > max_coord or abs(f.y) > max_coord:
            return False, "corrupted_coordinates"
        
    return True, "valid"


# ── Vectorized replay processing ────────────────────────────────────────────

def process_replay(in_replay: str | Replay, beatmap_idx: int = None, beatmap_info: list = None):
    """
    Parse .osr replay infos & features.
    
    Args:
        in_replay (str | Replay): the .osr path or Replay object
        beatmap_idx (int): for multi processing, idx of the beatmap on the zarr file init on _init_worker()
        beatmap_info (list): for single processing, contains [objects (ndarray), cs_raw (float), od_raw (float)]
    
    Returns:
        replay_hash (str): MD5 hash of the replay
        features (list): calculte replay features
        beatmap_hash (str): MD5 hash of the beatmap
        mods (int): Replay mods as mask, refer to osrparse.Mods
    """
    if isinstance(in_replay, str):
        replay = Replay.from_path(in_replay)
    elif isinstance(in_replay, Replay):
        replay = in_replay
    else:
        raise KeyError(f"Invalid in_replay type: {type(in_replay)}")

    if beatmap_info:
        if len(beatmap_info) < 3:
            raise KeyError("Invalid beatmap_info")
        
        objects_arr  = beatmap_info[0]
        bd = [BeatmapObject(*row) for row in objects_arr]
        cs_raw = beatmap_info[1]
        od_raw = beatmap_info[2]

    elif beatmap_idx: # On zarr file, multiprocessing
        beatmap_data = _worker_beatmap[beatmap_idx]
        objects_arr  = beatmap_data['objects']
        bd = [BeatmapObject(*row) for row in objects_arr]
        cs_raw = float(beatmap_data['cs'])
        od_raw = float(beatmap_data['od'])

        success, _ = validate_replay(replay, bd)
        if not success:
            return None
    
    else: # Invalid Args
        raise KeyError("Invalid args")

    r_data = replay.replay_data
    n = len(r_data)
    if n < 4:
        return None

    # -- Extract raw arrays from replay data --
    x        = np.empty(n, dtype=np.float32)
    y        = np.empty(n, dtype=np.float32)
    t        = np.empty(n, dtype=np.float32)
    keys_raw = np.empty(n, dtype=np.int32)
    for i, f in enumerate(r_data):
        x[i]        = f.x
        y[i]        = f.y
        t[i]        = f.time_delta
        keys_raw[i] = int(f.keys)

    # -- Velocity --
    dx = np.diff(x)
    dy = np.diff(y)
    dt = t[1:].copy()
    dt_safe = np.where(dt <= 0, 1.0, dt)

    dist = np.sqrt(dx * dx + dy * dy)
    v    = np.zeros(n, dtype=np.float32)
    v[1:] = dist / dt_safe

    # -- Acceleration --
    a = np.zeros(n, dtype=np.float32)
    a[2:] = (v[2:] - v[1:-1]) / dt_safe[1:]

    # -- Jerk --
    jerk = np.zeros(n, dtype=np.float32)
    jerk[3:] = (a[3:] - a[2:-1]) / dt_safe[2:]

    # -- Turning angle --
    angle  = np.zeros(n, dtype=np.float32)
    theta  = np.arctan2(dy, dx) # length n-1
    dtheta = np.diff(theta)     # length n-2
    angle[3:] = np.arctan2(np.sin(dtheta[1:]), np.cos(dtheta[1:]))

    # -- Keys --
    SMOKE = 16
    K1_M1 = 5    # Key.K1(4) | Key.M1(1)
    K2_M2 = 10   # Key.K2(8) | Key.M2(2)

    keys_clean = keys_raw & ~SMOKE
    key_a = ((keys_clean & K1_M1) != 0).astype(np.float32)
    key_b = ((keys_clean & K2_M2) != 0).astype(np.float32)


    pressed    = (keys_clean != 0)
    key_down   = np.zeros(n, dtype=np.float32)
    key_down[1:] = (pressed[1:] & ~pressed[:-1]).astype(np.float32)

    # -- Normalize --
    x_n     = x / 512.0
    y_n     = y / 384.0
    t_n     = t / 16.0
    v_n     = np.clip(v, 0, 10) / 10.0
    a_n     = np.clip(a, -5, 5) / 5.0
    jerk_n  = np.clip(jerk, -2, 2) / 2.0
    angle_n = angle / np.pi

    # -- Augment with beatmap target --
    cs, od = apply_mods_to_difficulty(cs_raw, od_raw, replay.mods)
    hit_radius = (54.4 - 4.48 * cs) / 512.0 # refer to: https://osu.ppy.sh/wiki/fr/Beatmap/Circle_size
    hit_window = (80.0 - 6.0 * od) / 16.0   # refer to: https://osu.ppy.sh/wiki/en/Beatmap/Overall_difficulty

    # Beatmap object arrays
    bd_x = objects_arr[:, 0].astype(np.float16)
    bd_y = objects_arr[:, 1].astype(np.float16)
    bd_t = objects_arr[:, 2].astype(np.float16)

    bd_cum     = np.cumsum(bd_t)
    replay_cum = np.cumsum(t_n)

    # For each frame, find the last beatmap object whose cumulative time <= frame time
    obj_indices = np.searchsorted(bd_cum, replay_cum, side='right') - 1
    obj_indices = np.clip(obj_indices, 0, len(bd_cum) - 1)

    target_x   = bd_x[obj_indices]
    target_y   = bd_y[obj_indices]
    target_cum = bd_cum[obj_indices]

    tdx = x_n - target_x
    tdy = y_n - target_y
    dist_to_target = np.sqrt(tdx * tdx + tdy * tdy) / hit_radius
    time_to_target = (target_cum - replay_cum) / hit_window
    click_offset   = np.where(
        key_down > 0,
        (replay_cum - target_cum) / hit_window,
        0.0
    )

    # -- Build output array (n, 13) --
    # WARNING: Because of vectorisation, array is not scalable
    out = np.column_stack([
        x_n, y_n, t_n, v_n, a_n, jerk_n, angle_n,
        key_a, key_b, key_down,
        dist_to_target, time_to_target, click_offset
    ]).astype(np.float16)

    return replay.replay_hash, out, replay.beatmap_hash, replay.mods


def get_hashes(path: str):
    with open(path, 'rb') as f:
        b_hash, r_hash = get_beatmap_replay_hash(f.read())
    return path, b_hash, r_hash

class PrepairReplays:
    def __init__(self):
        pass

    def process_folder(
        self, 
        in_replays: str, 
        beatmaps_zarr: str, 
        replays_zarr: str,
        max_workers: int = None,
        timeout:     float = 5.0
    ):
        assert os.path.exists(in_replays),  f"Folder {in_replays} doesn't exist"
        assert os.path.exists(beatmaps_zarr), f"Zarr {beatmaps_zarr} doesn't exist"
        
        os.makedirs(os.path.dirname(replays_zarr), exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        # Exclude to not demand hash every time to server
        exclude_path = os.path.join("temp", "exclude_beatmap.pkl")
        exclude = set()
        if os.path.exists(exclude_path):
            with open(exclude_path, 'rb') as f:
                exclude = pickle.load(f)
        
        
        paths = [(os.path.join(in_replays, filename),)
                 for filename in os.listdir(in_replays)]
        
        r_dataset = ZarrReplay(replays_zarr)
        b_dataset = ZarrBeatmap(beatmaps_zarr)

        pending_beatmap = []
        pending_replay  = []


        print("-- Get hashes --")
        with ProcessPool(max_workers=max_workers or os.cpu_count()) as pool:
            futures = {pool.schedule(get_hashes, args=path, timeout=timeout) for path in paths}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = fut.result()
                    if result is not None:
                        path, b_hash, r_hash = result
                        if b_hash in exclude:
                            continue
                        if not b_hash in b_dataset:
                            pending_beatmap.append((b_hash,))
                        if not r_hash in r_dataset:
                            pending_replay.append((path, b_hash))
                
                except TimeoutError:
                    pass
                except Exception as e:
                    pass
        del paths

        print("-- Download beatmaps --")
        with ProcessPool(max_workers=max_workers or os.cpu_count()) as pool:
            futures = {pool.schedule(download_beatmap, args=b_hash, timeout=timeout) for b_hash in pending_beatmap}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = fut.result()
                    if result is not None: 
                        if result[0] is not None:
                            b_dataset += result
                        else:
                            exclude.add(result[1])

                except TimeoutError:
                    pass
                except Exception as e:
                    pass
        
        del pending_beatmap
        del b_dataset
        b_dataset = ZarrBeatmap(beatmaps_zarr)
        
        # Save exclude
        with open(exclude_path, 'wb') as f:
            pickle.dump(exclude, f, protocol=pickle.HIGHEST_PROTOCOL)
        del exclude
        
        # Build args
        args = []
        b_dataset.build_hash_to_idx()
        for path, b_hash in pending_replay:
            idx = b_dataset.hash_to_idx.get(b_hash)
            if idx is not None:
                args.append((path, idx))

        del pending_replay
        
        print(" -- Process replay --")
        with ProcessPool(
            max_workers=max_workers or os.cpu_count(),
            initializer=_init_worker,
            initargs=(beatmaps_zarr,)
        ) as pool:
            futures = {pool.schedule(process_replay, args=a, timeout=timeout) for a in args}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = fut.result()
                    if result is not None:
                        r_dataset += result

                except TimeoutError:
                    pass
                except Exception as e:
                    pass
            r_dataset.flush_batch()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--replays-folder", required=True, type=str, help="Folder with replay (.osr) to proceed")
    parser.add_argument("--beatmap-zarr",   required=True, type=str, help="Proceed beatmap dataset (.zarr)")
    parser.add_argument("--out-path",       required=True, type=str, help="Process replays dataset path (.zarr)")
    parser.add_argument("--num-workers",    required=False, type=int, default=None)

    args = parser.parse_args()

    prepair = PrepairReplays()
    prepair.process_folder(
        args.replays_folder,
        args.beatmap_zarr,
        args.out_path,
        args.num_workers
    )
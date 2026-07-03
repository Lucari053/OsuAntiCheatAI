import os
import numpy as np
import requests
from zipfile import ZipFile
from pebble import ProcessPool
from concurrent.futures import TimeoutError as FutureTimeoutError, as_completed
from tqdm import tqdm
import argparse

from parser.beatmap_parser import BeatmapParser
from parser.slidercalc import SliderPath
from dataset.utils import get_beatmap_hash, BeatmapObject, ZarrBeatmap



def download_beatmap(md5_hash: str) -> bool:
    """
    Download beatmap by hash and extract objects & difficulty

    Returns:
        map_hash (str): the beatmap MD5 hash
        objects (list[float]): BeatmapObject as array
        cs (float): circle size difficulty
        od (float): overall dificulty
    """
    meta_url = f"https://akatsuki.gg/api/get_beatmaps?h={md5_hash}"

    try:
        # Seach beatmap ID
        response = requests.get(meta_url, timeout=5)
        if response.status_code != 200 or not response.json():
            # print("ERROR: Can't access ID server")
            return None, md5_hash

        beatmap_data = response.json()[0]
        beatmap_id   = beatmap_data["beatmap_id"]
        download_url = f"https://osu.ppy.sh/osu/{beatmap_id}"

        
        file_response = requests.get(download_url)
        
        if file_response.status_code == 200:
            return process_beatmap_content(file_response.content)
        else:
            print("ERROR: Can't access download server")
            return None, md5_hash
    
    except Exception as e:
        print(f"ERROR: {e}")
        return None, md5_hash

def build_beatmap_objects(beatmap: dict) -> list:
    
    final = []
    prev_time = 0
    
    for obj in beatmap['hitObjects']:
        
        # Slider
        match obj['object_name']:
            case "slider":
                final.append(BeatmapObject(*obj['position'], obj['startTime'] - prev_time, 1))

                last_t = obj['startTime']
                for sx, sy, s_abs_t in sample_slider_points(obj):
                    final.append(BeatmapObject(sx, sy, s_abs_t - last_t, 2))
                    last_t = s_abs_t
                
                final.append(BeatmapObject(*obj['end_position'], obj['end_time'] - last_t, 3))
                prev_time = obj['end_time']

            case "spinner":
                final.append(BeatmapObject(*obj['position'], obj['startTime'] - prev_time, 4))
                final.append(BeatmapObject(*obj['position'], obj['end_time'] - obj['startTime'], 4))
                prev_time = obj['end_time']
            
            case _: # Circle
                final.append(BeatmapObject(*obj['position'], obj['startTime'] - prev_time, 0))
                prev_time = obj['startTime']
    
    return final
        

def sample_slider_points(obj: dict, sample_dt: float = 16.0):
    duration = obj.get("duration")
    if not duration or duration <= 0:
        return []

    duration = min(duration, 120_000) # Beucase of corrupted data
    
    repeat_count         = obj["repeatCount"]
    single_pass_duration = duration / repeat_count
    pixel_length         = obj["pixelLength"]
    curve_type           = obj.get("curveType", "linear")
    points               = obj["points"]

    path = obj.get("slider_path") or SliderPath(curve_type, points)

    samples = []
    t = sample_dt
    while t < duration:
        pass_idx = int(t // single_pass_duration)
        local_t = t - pass_idx * single_pass_duration
        dist = (local_t / single_pass_duration) * pixel_length
        if pass_idx % 2 == 1:
            dist = pixel_length - dist
        
        point = path.point_at_distance(dist)
        if point and abs(point[0]) < 1e5 and abs(point[1]) < 1e5:
            samples.append((point[0], point[1], obj["startTime"] + t))

        t += sample_dt

    return samples

def process_beatmap_content(byte_content: bytes):
    """
    Parse beatmap byte content and extract objects & difficulty

    Returns:
        map_hash (str): the beatmap MD5 hash
        objects (list[float]): BeatmapObject as array
        cs (float): circle size difficulty
        od (float): overall dificulty
    """
    map_hash = get_beatmap_hash(byte_content)

    parser = BeatmapParser()
    for line in byte_content.decode().split("\n"):
        parser.read_line(line)

    beatmap = parser.build_beatmap()
    if beatmap['Mode'] != "0": return # Only osu std support

    cs = float(beatmap.get("CircleSize", 4))
    od = float(beatmap.get("OverallDifficulty", 5))
    
    objects = build_beatmap_objects(beatmap)
    final = []
    for obj in objects:
        obj.normalize()
        final.append(obj.array())
    
    final = np.array(final, dtype=np.float16)
    return map_hash, final, cs, od



def process_beatmap_osu(path: str) -> list:
    """
    Parse .osu beatmap and extract objects & difficulty

    Returns:
        map_hash (str): the beatmap MD5 hash
        objects (list[float]): BeatmapObject as array
        cs (float): circle size difficulty
        od (float): overall dificulty
    """
    with open(path, 'rb') as f:
        byte_content = f.read()

    try:
        return process_beatmap_content(byte_content)
    except Exception as e:
        # print(f"Beatmap error: {e}")
        pass


def process_beatmap_osz(path: str, out_dir: str):

    # unzip .osz file
    with ZipFile(path, 'r') as zObject:
        # Get beatmap list
        beatmaps = [name for name in zObject.namelist() 
                   if name.split(".")[-1] == "osu"
        ]
        for b in beatmaps:

            # Bytes content
            with zObject.open(b) as f:
                byte_content = f.read()

            return process_beatmap_content(byte_content)


def get_osu_hash(path: str):
    with open(path, 'rb') as f:
        b_hash = get_beatmap_hash(f.read())
    return path, b_hash


class PrepairBeatmap:
    def __init__(self):
        pass

    def process_folder(self, beatmaps_path: str, out_path: str, max_workers: int = None, timeout=2):

        assert os.path.exists(beatmaps_path) or os.path.isdir(beatmaps_path), f"Folder {beatmaps_path} doesn't exist"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        dataset = ZarrBeatmap(out_path) # Create dataset

        paths = [(os.path.join(beatmaps_path, filename),)
                for filename in os.listdir(beatmaps_path)]
        
        # Get all valid hash
        args = []
        print("-- Load dataset --")
        with ProcessPool(max_workers=max_workers or os.cpu_count()) as pool:
            futures = {pool.schedule(get_osu_hash, args=path) for path in paths} 
            
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = fut.result()
                    if result is not None:
                        path, b_hash = result
                        if not b_hash in dataset:
                            args.append((path,))
                except FutureTimeoutError:
                    pass
                except Exception as e:
                    print(e)
        
        # Proceed all osu beatmap
        print("-- Proceed beatmap --")
        with ProcessPool(max_workers=max_workers or os.cpu_count()) as pool:
            futures = {pool.schedule(process_beatmap_osu, args=a, timeout=timeout) for a in args}
            
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result = fut.result()
                    if result is not None:
                        dataset += result
                
                except FutureTimeoutError:
                    pass
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--beatmaps-folder", required=True, type=str, help="Folder that containing beatmap (.osu) to proceed")
    parser.add_argument("--out-path",        required=True, type=str, help="Process beatmap dataset path (.zarr)")
    parser.add_argument("--num-workers",     required=False, default=None, type=int)

    args = parser.parse_args()

    prepair = PrepairBeatmap()
    prepair.process_folder(
        args.beatmaps_folder, 
        args.out_path, 
        args.num_workers
    )
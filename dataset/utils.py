import hashlib
from dataclasses import dataclass, astuple
import math
import zarr
import numpy as np
import os

def clamp(minimum, maximum, val):
    return max(minimum, min(maximum, val))

def get_beatmap_hash(content: bytes):
    return hashlib.md5(content).hexdigest()

@dataclass
class ReplayFrame:
    x:     float = 0.0
    y:     float = 0.0
    t:     float = 0.0
    v:     float = 0.0
    a:     float = 0.0
    jerk:  float = 0.0
    angle: float = 0.0
    key_a: float = 0.0
    key_b: float = 0.0
    key_down: float = 0.0
    dist_to_target: float = 0.0
    time_to_target: float = 0.0
    click_offset:   float = 0.0

    def normalize(self):
        self.x     /= 512
        self.y     /= 384
        self.t     /= 16
        self.v      = clamp(0, 10, self.v)    / 10
        self.a      = clamp(-5, 5, self.a)    / 5
        self.jerk   = clamp(-2, 2, self.jerk) / 2
        self.angle /= math.pi
 

    def array(self):
        return list(astuple(self))
    
    def is_pressed(self) -> bool:
        return bool(self.key_a or self.key_b)

    def __len__(self):
        return len(astuple(self))
    
@dataclass
class BeatmapObject:
    x: float = 0.0
    y: float = 0.0
    t: float = 0.0
    object_type: float = 0.0

    def normalize(self):
        self.x /= 512
        self.y /= 384
        self.t /= 16
        self.object_type /= 4

    def array(self):
        return list(astuple(self))
    
    def __len__(self):
        return len(astuple(self))

@dataclass
class Pairs:
    beatmap_start: int = 0 # Circle start of the beatmap
    beatmap_count: int = 0 
    replay_start:  int = 0 # Frame start of the replay
    replay_count:  int = 0 

    def array(self):
        return list(astuple(self))
    
    def __len__(self):
        return len(astuple(self))


class Zarr:
    def __init__(self):
        self.bhash_to_idx = False
        self.hash_to_idx  = None
        self.z_hash       = []
        self.z_indptr     = []
        
        self.processed_hashes = set()

    def build_hash_to_idx(self):
        self.bhash_to_idx = True
        self.hash_to_idx = {h: idx for idx, h in enumerate(self.z_hash[:])}

    def __getitem__(self, key):
        if isinstance(key, str): # str has beatmap hash
            if not self.bhash_to_idx or key not in self.hash_to_idx:
                raise KeyError(f"Hash {key} does not exist in dataset")
            
            idx = self.hash_to_idx[key]
        
        elif isinstance(key, int): # int has idx
            if self.z_indptr.shape[0] > key:
                idx = key
            else:
                raise ValueError(f"Idx {key} out of range")
        
        else: # else type errore
            raise TypeError(f"Invalid key type {key}")
        
        return idx
    
    def __contains__(self, r_hash: str) -> bool:
        return r_hash in self.processed_hashes
    
    def __iadd__(self, other):
        if self.bhash_to_idx:
            self.hash_to_idx[other] = len(self.z_hash)



class ZarrBeatmap(Zarr):
    def __init__(self, path: str):
        super().__init__()

        exist = os.path.exists(path)

        store = zarr.DirectoryStore(path)
        self.root = zarr.group(store)

        

        if not exist:
            chunks_size = 10_000
            obj_len = len(BeatmapObject())
            self.z_object = self.root.create_dataset('objects', shape=(0, obj_len), chunks=(500_000, obj_len), dtype='float16')
            self.z_hash   = self.root.create_dataset('hash',    shape=(0,),         chunks=(chunks_size,),         dtype='str')
            self.z_cs     = self.root.create_dataset('cs',      shape=(0,),         chunks=(chunks_size,),         dtype='float16')
            self.z_od     = self.root.create_dataset('od',      shape=(0,),         chunks=(chunks_size,),         dtype='float16')
            self.z_indptr = self.root.create_dataset('indptr',  shape=(1,),         chunks=(chunks_size,),         dtype='int64', data=[0])

            self.current_ptr = 0
        else:
            self.z_object = self.root['objects']
            self.z_hash   = self.root['hash']
            self.z_cs     = self.root['cs']
            self.z_od     = self.root['od']
            self.z_indptr = self.root['indptr']
            self.processed_hashes = set(self.z_hash[:])

            self.current_ptr = int(self.z_indptr[-1])
        
    def __getitem__(self, key: str | int):
        idx = super().__getitem__(key)

        start = self.z_indptr[idx]
        end = self.z_indptr[idx + 1]

        objects = self.z_object[start:end]
        cs = self.z_cs[idx]
        od = self.z_od[idx]

        return {
            "objects": objects,
            "cs": cs,
            "od": od,
            "idx": idx
        }


    def __iadd__(self, other: list):
        """
        Args:
            other (list): (map_hash, objects_arr, cs, od)
        """
        
        map_hash, objects_arr, cs, od = other

        super().__iadd__(map_hash)
        # Don't allow same hash
        if map_hash in self:
            return self

        self.z_object.append(objects_arr)
        self.z_hash.append(np.array([map_hash], dtype='str'))
        self.z_cs.append(np.array([cs], dtype='float16'))
        self.z_od.append(np.array([od], dtype='float16'))

        self.current_ptr += len(objects_arr)
        self.z_indptr.append(np.array([self.current_ptr], dtype='int64'))
        
        self.processed_hashes.add(map_hash)
        
        return self


class ZarrReplay(Zarr):
    def __init__(self, path: str, batch_size: int = 500):
        super().__init__()
        exist = os.path.exists(path)
        
        store = zarr.DirectoryStore(path)
        self.root = zarr.group(store)

        if not exist:
            chunks_size = 10_000
            obj_len = len(ReplayFrame())
            self.z_data    = self.root.create_dataset('data',    shape=(0, obj_len), chunks=(100_000, obj_len), dtype='float16')
            self.z_hash    = self.root.create_dataset('hash',    shape=(0,),         chunks=(chunks_size,),         dtype='str')
            self.z_beatmap = self.root.create_dataset('beatmap', shape=(0,),         chunks=(chunks_size,),         dtype='str')
            self.z_mods    = self.root.create_dataset('mods',    shape=(0,),         chunks=(chunks_size,),         dtype='int32')
            self.z_indptr  = self.root.create_dataset('indptr',  shape=(1,),         chunks=(chunks_size,),         dtype='int64', data=[0])

            self.current_ptr = 0
        else:
            self.z_data    = self.root['data']
            self.z_hash    = self.root['hash']
            self.z_beatmap = self.root['beatmap']
            self.z_mods    = self.root['mods']
            self.z_indptr  = self.root['indptr']
            self.processed_hashes = set(self.z_hash[:])

            self.current_ptr = int(self.z_indptr[-1])
        
        # Batch write buffers
        self._batch_data = []
        self._batch_hash = []
        self._batch_beatmap = []
        self._batch_mods = []
        self._batch_size = batch_size

    def __getitem__(self, key: str | int):
        
        idx = super().__getitem__(key)

        start = self.z_indptr[idx]
        end = self.z_indptr[idx + 1]

        datas   = self.z_data[start:end]
        r_hash  = self.z_hash[idx]
        beatmap = self.z_beatmap[idx]
        mods    = self.z_mods[idx]

        return {
            "datas":   datas,
            "hash":    r_hash,
            "beatmap": beatmap,
            "mods":    mods,
            "idx":     idx
        }

    def __iadd__(self, other: list):
        """
        Args:
            other (list): (replay_hash, frame_arr, beatmap_hash, mods)
        """
        
        replay_hash, frame_arr, beatmap_hash, mods = other
        super().__iadd__(replay_hash)

        # Don't allow same hash
        if replay_hash in self:
            return self

        self._batch_data.append(frame_arr)
        self._batch_hash.append(replay_hash)
        self._batch_beatmap.append(beatmap_hash)
        self._batch_mods.append(int(mods))
        self.processed_hashes.add(replay_hash)

        if len(self._batch_hash) >= self._batch_size:
            self.flush_batch()

        return self

    def flush_batch(self):
        """Flush accumulated batch, single write."""
        if not self._batch_hash:
            return

        all_data = np.concatenate(self._batch_data)
        self.z_data.append(all_data)
        self.z_hash.append(np.array(self._batch_hash, dtype='str'))
        self.z_beatmap.append(np.array(self._batch_beatmap, dtype='str'))
        self.z_mods.append(np.array(self._batch_mods, dtype='int32'))

        new_ptrs = []
        for arr in self._batch_data:
            self.current_ptr += len(arr)
            new_ptrs.append(self.current_ptr)
        self.z_indptr.append(np.array(new_ptrs, dtype='int64'))

        self._batch_data.clear()
        self._batch_hash.clear()
        self._batch_beatmap.clear()
        self._batch_mods.clear()


class ZarrPair(Zarr):
    def __init__(self, path: str, batch_size: int = 1000):
        super().__init__()
        exist = os.path.exists(path)

        store = zarr.DirectoryStore(path)
        self.root = zarr.group(store)

        if not exist:
            chunks_size = 10_000
            pair_len = len(Pairs())
            self.z_pair    = self.root.create_dataset('pair',    shape=(0, pair_len), chunks=(100_000, pair_len), dtype='int32')
            self.z_hash    = self.root.create_dataset('hash',    shape=(0,),          chunks=(chunks_size,),      dtype='str')
            self.z_beatmap = self.root.create_dataset('beatmap', shape=(0,),          chunks=(chunks_size,),      dtype='str')
            self.z_indptr  = self.root.create_dataset('indptr',  shape=(1,),          chunks=(chunks_size,),      dtype='int64', data=[0])

            self.current_ptr = 0
        else:
            self.z_pair    = self.root['pair']
            self.z_hash    = self.root['hash']
            self.z_beatmap = self.root['beatmap']
            self.z_indptr  = self.root['indptr']
            self.processed_hashes = set(self.z_hash[:])

            self.current_ptr = int(self.z_indptr[-1])
        
        # Batch write buffers
        self._batch_pair = []
        self._batch_hash = []
        self._batch_beatmap = []
        self._batch_size = batch_size
        

    def __getitem__(self, key: str | int):
        
        idx = super().__getitem__(key)

        start = self.z_indptr[idx]
        end = self.z_indptr[idx + 1]

        pairs = self.z_pair[start:end]
        r_hash  = self.z_hash[idx]
        beatmap = self.z_beatmap[idx]

        return {
            "pairs":   pairs,
            "hash":    r_hash,
            "beatmap": beatmap,
            "idx":     idx
        }

    def __iadd__(self, other: list):
        """
        Args:
            other (list): (replay_hash, pair, beatmap_hash)
        """
        
        replay_hash, pair_arr, beatmap_hash = other
        super().__iadd__(replay_hash)

        # Don't allow same hash
        if replay_hash in self:
            return self

        self._batch_pair.append(pair_arr)
        self._batch_hash.append(replay_hash)
        self._batch_beatmap.append(beatmap_hash)
        self.processed_hashes.add(replay_hash)

        if len(self._batch_hash) >= self._batch_size:
            self.flush_batch()
        
        return self

    def flush_batch(self):
        """Flush accumulated batch of pairs to Zarr storage in single write per array."""
        if not self._batch_hash:
            return

        all_pairs = np.concatenate(self._batch_pair)
        self.z_pair.append(all_pairs)
        self.z_hash.append(np.array(self._batch_hash, dtype='str'))
        self.z_beatmap.append(np.array(self._batch_beatmap, dtype='str'))

        new_ptrs = []
        for arr in self._batch_pair:
            self.current_ptr += len(arr)
            new_ptrs.append(self.current_ptr)
        self.z_indptr.append(np.array(new_ptrs, dtype='int64'))

        self._batch_pair.clear()
        self._batch_hash.clear()
        self._batch_beatmap.clear()
import os
import torch
import numpy as np
from osu import Client
from osrparse import Replay
from enum import Enum
from dotenv import load_dotenv

from dataset.prepaire_beatmap import process_beatmap_osu, download_beatmap
from dataset.prepaire_replay import process_replay
from dataset.prepaire_pairs import prepair_pairs
from model.model import CheatDetector
from config import Config


load_dotenv()

class LoadReplayOutput(Enum):
    SUCCESS      = "SUCCESS"
    INVALID_PATH = "INVALID PATH"
    NEED_BEATMAP = "NEED BEATMAP"
    FAIL         = "FAIL"

def create_aligned_windows(beatmap_objects: np.ndarray, replay_features: np.ndarray, max_windows: int = 300):
    """
    Align replay & beatmap objects into fixed context windows (pairs).
    """
    pairs = prepair_pairs(replay_features, beatmap_objects)

    if max_windows is not None and len(pairs) > max_windows:
        idxs = np.linspace(0, len(pairs) - 1, max_windows, dtype=int)
        pairs = pairs[idxs]
    
    beatmap_windows = np.stack([beatmap_objects[b0:b0 + bc] for b0, bc, _, _ in pairs])
    replay_windows = np.stack([replay_features[r0:r0 + rc] for _, _, r0, rc in pairs])

    # Dynamic dataset normalisation
    beatmap_windows[:, :, 2] = np.clip(beatmap_windows[:, :, 2], 0.0, 50.0) / 50.0
    replay_windows[:, :, 2] = np.clip(replay_windows[:, :, 2], 0.0, 50.0) / 50.0
    replay_windows[:, :, 10] = np.clip(replay_windows[:, :, 10], 0.0, 10.0) / 10.0
    replay_windows[:, :, 11] = np.clip(replay_windows[:, :, 11], -10.0, 10.0) / 10.0
    replay_windows[:, :, 12] = np.clip(replay_windows[:, :, 12], -10.0, 10.0) / 10.0
    
    return (
        torch.tensor(beatmap_windows, dtype=torch.float32),
        torch.tensor(replay_windows, dtype=torch.float32)
    )

class CheatDetectorInfer:
    def __init__(self, checkpoint_path: str):

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not fount at {checkpoint_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        d_model = Config.get("model/d_model")
        self.max_windows = Config.get("train/max_windows")

        # Load model
        self.model = CheatDetector(d_model).to(self.device)

        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        state_dict = (checkpoint_data['model_state_dict'] 
                      if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data 
                      else checkpoint_data)
        
        # Strip compilation prefix if present
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                clean_state_dict[k[10:]] = v
            else:
                clean_state_dict[k] = v

        self.model.load_state_dict(clean_state_dict)
        self.model.eval()

        # Setup client download
        self.client = None
        try:
            self.client = Client.from_credentials(
                os.environ["CLIENT_ID"],
                os.environ["CLIENT_SECRET"],
                os.environ["REDIRECT_URL"] or 'http://localhost:8080'
            )
        except Exception as e:
            pass
    
    def load_replay(self, val: str)-> bool:
        """
        Args:
            val (str): .osr path OR replay_link OR score_id
        """
        # -- Get replay as osrparse --
        # .osr path
        if os.path.exists(val) and os.path.splitext(val)[-1] == ".osr":
            self.replay = Replay.from_path(val)
            
        # score link/id
        elif val.split("/")[-1].isdigit():
            score_id = int(val.split("/")[-1])

            if self.client is None:
                raise KeyError("""To use replay_link/score_id, you need to refer in a .env file:
                               - CLIENT_ID, - CLIENT_SECRET""")

            self.replay: Replay = self.client.get_replay_data_by_id_only(
                score_id=score_id,
                use_osrparse=True
            )

        # Invalid
        else:
            raise KeyError(f"Invalid load_replay() arg {val}")

        
        if self.replay is None:
            return LoadReplayOutput.INVALID_PATH
        
        # Process replay
        result = download_beatmap(self.replay.beatmap_hash)
        if result is None or result[0] is None:
            print(f"Invalid betmap hash, please refer .osu beatmap on load_beatmap()")
            return LoadReplayOutput.NEED_BEATMAP
        
        _, self.beatmap_objects, cs, od = result
        _, self.replay_features, _, _ = process_replay(self.replay, beatmap_info=[self.beatmap_objects, cs, od])
        return LoadReplayOutput.SUCCESS
        
    
    def load_beatmap(self, beatmap_path: str):
        """
        Load the beatmap, need to be the same as replay (load_replay())
        Args:
            beatmap_path (str): the .osu beatmap path
        """
        if not os.path.exists(beatmap_path):
            raise FileNotFoundError(f"beatmap_path not found at {beatmap_path}")
        
        beatmap_hash, self.beatmap_objects, cs, od = process_beatmap_osu(beatmap_path)

        if not self.replay or self.replay.beatmap_hash != beatmap_hash:
            raise KeyError(f"Beatmap is not the same as replay beatmap. Please set load_replay() before.")
        
        _, self.replay_features, _, _ = process_replay(self.replay, beatmap_info=[self.beatmap_objects, cs, od])
        
    def infer(self):

        # Align replay/beatmap
        bm_t, rp_t = create_aligned_windows(self.beatmap_objects, self.replay_features, self.max_windows)

        # Add batch dimension
        bm_t = bm_t.unsqueeze(0).to(self.device) # [1, num_windows, n_objects, n_feat_b]
        rp_t = rp_t.unsqueeze(0).to(self.device) # [1, num_windows, n_frame, n_feat_r]
        mask = torch.zeros(1, bm_t.shape[1], dtype=torch.bool).to(self.device) # No mask needed for batch size 1

        device_str = "cuda" if self.device == torch.device("cuda") else "cpu"
        with torch.no_grad():
            with torch.autocast(device_type=device_str, dtype=torch.bfloat16):
                logits, _ = self.model(bm_t, rp_t, key_padding_mask=mask)
                prob = torch.sigmoid(logits).item()

        return prob >= 0.5, prob
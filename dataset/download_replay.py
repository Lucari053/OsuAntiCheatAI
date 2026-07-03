"""
WARNING: to slow in test, same with multiprocessing
You can use this instead: https://www.kaggle.com/datasets/skihikingkevin/ordr-replay-dump
"""

import os
from osu import Client, UserScoreType, SoloScore
import random
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

client = Client.from_credentials(
    os.environ["CLIENT_ID"],
    os.environ["CLIENT_SECRET"],
    os.environ["REDIRECT_URL"],
    limit_per_minute=100000,
    request_wait_time=0.1
)

def download_score(score: SoloScore, out_folder: str) -> bool:
    
    try:
        if not score.replay: return False # No replay available
        
        out_path = os.path.join(out_folder, f"{score.id}.osr")
        
        # Check already exist
        if os.path.exists(out_path): return False

        # Download replay as bytes data
        replay_data = client.get_replay_data_by_id_only(
            score_id=score.id,
            use_osrparse=False
        )
        
        if not replay_data:
            return False
        
        # Write file
        with open(out_path, 'wb') as f:
            f.write(replay_data)
        
        return True
    
    except Exception as e:
        return False
    
def download_user(user_id, out_folder) -> int:
    
    added = 0
    try:
        for score_type in [UserScoreType.BEST, UserScoreType.PINNED, UserScoreType.RECENT]:

            user_scores = client.get_user_scores(
                user_id, 
                score_type,
                mode="osu",
                limit=None
            )

            for score in user_scores:
                if type(score) != SoloScore:
                    continue

                success = download_score(score, out_folder)
                if success:
                    added += 1
        
        return added
    
    except Exception as e:
        return added




class DownloadReplay:
    def download_random(self, out_folder: str, limit: int = 100):
        
        os.makedirs(out_folder, exist_ok=True)

        process = 0
        pbar = tqdm(total=limit, unit="replay")
        while process < limit:

            user_id = random.randint(100_000, 20_000_000)
            added = download_user(user_id, out_folder)

            limit += added
            pbar.update(added)


if __name__ == "__main__":
    p = DownloadReplay()

    p.download_random("out/download_replay", 2)
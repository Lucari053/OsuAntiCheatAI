# osu! std AI Cheat Detector

The AI Cheat-Detector for osu! std is a Transformer-based sequence classification system design to detect cheats (including **Relax Hacks**, **Aim Assist**, **Aim Correction** and **Human Autobot**) that take replay frames and beatmap objects.

> [!IMPORTANT]  
> The AI can generate false response and is not 100% accurate

## Setup & Installation
1. **Clone the Repository:**
    ```bash
   git clone https://github.com/yourusername/OsuAntiCheat.git
   cd OsuAntiCheat
    ```

2. **Install Dependencies:**
    ```bash
   pip install -r requirements.txt
    ```
3. **Configure Environment Variables (Optional):**
   
    If you plan to use score_id or link from the official osu! API, create a `.env` file in the root directory:
    ```env
    CLIENT_ID=your_client_id
    CLIENT_SECRET=your_client_secret
    REDIRECT_URL=http://localhost
    ```

## Prepare data
1. **Download beatmaps & replays**

    I recommand to use the [ordr dataset from skihikingkevin](https://www.kaggle.com/datasets/skihikingkevin/ordr-replay-dump), it offer 300k+ replays with beatmap.

    For cheating gameplay, you can generate it with
    ```bash
    python -m dataset.cheat_synth `
    --replays-folder  "{replays_folder}" `
    --beatmaps-folder "{beatmaps_folder}" `
    --output-folder   "out/cheat_osr"
    ```
2. **Process beatmaps**

    ```bash
    python -m dataset.prepaire_beatmap `
    --beatmaps-folder "{beatmaps_folder}" `
    --out-path        "out/beatmap.zarr"
    ```

3. **Process replays**

    Legit:
    ```bash
    python -m dataset.prepaire_replay `
    --replays-folder "{replays_folder}" `
    --beatmap-zarr   "out/beatmap.zarr" `
    --out-path       "out/replay_legit.zarr"
    ```
    Cheat:
    ```bash
    python -m dataset.prepaire_replay `
    --replays-folder "out/cheat_osr" `
    --beatmap-zarr   "out/beatmap.zarr" `
    --out-path       "out/replay_cheat.zarr"
    ```

4. **Create pairs**

    If you want to change window size, check the [config](configs/config.yaml) and edit the data section.

    Make label:
    ```bash
    python -m dataset.cheat_label_build `
    --replay-legit-zarr "out/replay_legit.zarr" `
    --replay-cheat-zarr "out/replay_cheat.zarr" `
    --out-label-json    "out/label.json" `
    --limit-per-label   100000
    ```
    Make pairs:
    ```bash
    python -m dataset.prepaire_pairs `
    --label-json        "out/label.json" `
    --replay-legit-zarr "out/replay_legit.zarr" `
    --replay-cheat-zarr "out/replay_cheat.zarr" `
    --beatmap-zarr      "out/beatmap.zarr" `
    --out-pairs-zarr    "out/pairs.zarr"
    ```

## How to train

1. Process the dataset with previous steps.
2. Check [config](configs/config.yaml)
3. Train
    ```bash
    python -m train `
    --pairs-zarr        "out/pairs.zarr" `
    --replay-legit-zarr "out/replay_legit.zarr" `
    --replay-cheat-zarr "out/replay_cheat.zarr" `
    --beatmap-zarr      "out/beatmap.zarr" `
    --label-json        "out/label.json" `
    --checkpoint-path   "out/checkpoints/best.pt"
    ```

## How to run inference
    
```bash
    python app.py
```
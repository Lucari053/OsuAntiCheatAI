import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import subprocess
import argparse

from dataset.dataset import ReplaySequenceDataset, collate_replays
from dataset.split import split_replays
from model.model import CheatDetector
from config import Config


def init_msvc_compiler_windows(compiler_path: str):
    if os.name != "nt": # Check windows
        return
    # Check if already configure
    if shutil.which("cl") and "INCLUDE" in os.environ:
        return
    
    if not compiler_path or not os.path.exists(compiler_path):
        return

    try:
        cmd = f'"{compiler_path}" && set'
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if '=' in  line:
                    key, val = line.split("=", 1)
                    if key.upper() in ["PATH", "INCLUDE", "LIB", "LIBPATH"] or key.startswith("VC") or key.startswith("VS"):
                        os.environ[key] = val
    
    except Exception as e:
        return

def model_size(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def model_size_as_string(size: int) -> str:
    if size < 1000:
        return str(size)
    
    POWER = ["", "K", "M", "B", "T"]
    power_idx = 0
    
    val = float(size)
    while val >= 1000:
        val /= 1000
        power_idx += 1

    return f"{val:.1f}".rstrip("0").rstrip(".") + POWER[power_idx]

def compute_metrics(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    # Compare model prediction to reality
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def run_epoch(
    model: CheatDetector, 
    loader: DataLoader, 
    criterion: nn.BCEWithLogitsLoss, 
    device, 
    optimizer=None
):
    device_str = "cuda" if device == torch.device("cuda") else "cpu"
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_logits, all_labels, total_loss = [], [], 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for beatmap_b, replay_b, mask, labels in tqdm(loader, leave=False):
            beatmap_b, replay_b, mask, labels = (
                beatmap_b.to(device), replay_b.to(device), mask.to(device), labels.to(device)
            )

            with torch.autocast(device_type=device_str, dtype=torch.bfloat16):
                logits, _ = model(beatmap_b, replay_b, key_padding_mask=mask)
                loss      = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_logits, all_labels)
        metrics["loss"] = total_loss / len(all_labels)
        return metrics
            
def train(
    pairs_folder:    str,
    legit_folder:    str,
    cheat_folder:    str,
    beatmaps_folder: str,
    labels_path:     str,
    checkpoint_path: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    n_epochs = Config.get("train/n_epochs")
    init_msvc_compiler_windows(
        compiler_path=r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    )

    train_hashes, val_hashes = split_replays(labels_path, Config.get("train/val_ratio"))

    train_ds = ReplaySequenceDataset(
        pairs_folder, legit_folder, cheat_folder,
        beatmaps_folder, labels_path, allowed_replays=train_hashes,
        max_windows=Config.get("train/max_windows")
    )
    val_ds = ReplaySequenceDataset(
        pairs_folder, legit_folder, cheat_folder,
        beatmaps_folder, labels_path, allowed_replays=val_hashes,
        max_windows=Config.get("train/max_windows")
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.get("train/batch_size"), 
        shuffle=True,
        collate_fn=collate_replays,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    val_loader   = DataLoader(
        val_ds,   
        batch_size=Config.get("train/batch_size"), 
        shuffle=False,
        collate_fn=collate_replays,
        pin_memory=True,
        num_workers=0
    )
    
    n_legits, n_cheat = train_ds.class_counts()
    print(f"Train: {n_legits:.0f} legit / {n_cheat:.0f} cheat")
    pos_weight = torch.tensor([n_legits / max(n_cheat, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model     = CheatDetector(Config.get("model/d_model")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(Config.get("train/lr")), weight_decay=1e-2)

    warmup_step = Config.get("train/warmup_step")
    total_step  = n_epochs * len(train_loader)

    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_step)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_step - warmup_step)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_step]
    )

    # Resume checkpoint if enabled
    start_epoch = 0
    global_step = 0
    best_f1 = 0.0
    resume_path = Config.get("train/resume")
    if resume_path:
        checkpoint_file = resume_path if isinstance(resume_path, str) else checkpoint_path.replace("best.pt", "last.pt")
        if os.path.exists(checkpoint_file):
            print(f"Resuming training from checkpoint: {checkpoint_file}")
            checkpoint_data = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Adjust T_max of the cosine scheduler in the loaded state_dict if n_epochs has changed
            if 'scheduler_state_dict' in checkpoint_data:
                try:
                    new_T_max = total_step - warmup_step

                    if '_schedulers' in checkpoint_data['scheduler_state_dict'] and len(checkpoint_data['scheduler_state_dict']['_schedulers']) > 1:
                        checkpoint_data['scheduler_state_dict']['_schedulers'][1]['T_max'] = new_T_max
                        print(f"Adjusted scheduler T_max to {new_T_max} based on new config n_epochs={n_epochs}")
                except Exception as e:
                    print(f"Warning adjusting scheduler T_max: {e}")
            
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            start_epoch = checkpoint_data['epoch']
            global_step = checkpoint_data.get('global_step', 0)
            best_f1 = checkpoint_data.get('best_f1', 0.0)
            print(f"Resumed from epoch {start_epoch+1}, step {global_step} (best validation F1: {best_f1:.3f})")
        else:
            print(f"No checkpoint found at {checkpoint_file}, starting from scratch.")

    model = torch.compile(model)
    print(f"Model size: {model_size_as_string(model_size(model))}")

    val_every_steps = Config.get("train/val_every_steps") or 1000
    log_every_steps = Config.get("train/log_every_steps") or 50

    running_loss = 0.0
    running_logits = []
    running_labels = []

    for epoch in range(start_epoch, n_epochs):
        model.train()
        device_str = "cuda" if device == torch.device("cuda") else "cpu"
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for beatmap_b, replay_b, mask, labels in pbar:
            beatmap_b, replay_b, mask, labels = (
                beatmap_b.to(device), replay_b.to(device), mask.to(device), labels.to(device)
            )

            with torch.autocast(device_type=device_str, dtype=torch.bfloat16):
                logits, _ = model(beatmap_b, replay_b, key_padding_mask=mask)
                loss      = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # Accumulate running stats
            running_loss += loss.item() * labels.size(0)
            running_logits.append(logits.detach().cpu())
            running_labels.append(labels.detach().cpu())

            # Periodic logging
            if global_step % log_every_steps == 0:
                running_logits_cat = torch.cat(running_logits)
                running_labels_cat = torch.cat(running_labels)
                train_metrics = compute_metrics(running_logits_cat, running_labels_cat)
                train_loss = running_loss / len(running_labels_cat)

                pbar.set_postfix({
                    "loss": f"{train_loss:.4f}",
                    "f1": f"{train_metrics['f1']:.3f}",
                    "acc": f"{train_metrics['accuracy']:.3f}"
                })

                # Reset running stats
                running_loss = 0.0
                running_logits = []
                running_labels = []

            # Periodic validation
            if global_step % val_every_steps == 0:
                print(f"\n[Step {global_step}] Running validation...")
                val_metrics = run_epoch(model, val_loader, criterion, device)
                model.train() # Restore training mode after run_epoch
                
                print(
                    f"[Step {global_step}] "
                    f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.3f} "
                    f"val_precision={val_metrics['precision']:.3f} val_recall={val_metrics['recall']:.3f}"
                )

                # Log validation to file
                try:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    val_log_path = os.path.join(os.path.dirname(checkpoint_path), "val_log.txt")
                    with open(val_log_path, "a") as f:
                        f.write(
                            f"[{timestamp}] Epoch {epoch+1}, Step {global_step}: "
                            f"loss={val_metrics['loss']:.4f}, f1={val_metrics['f1']:.3f}, "
                            f"precision={val_metrics['precision']:.3f}, recall={val_metrics['recall']:.3f}, "
                            f"accuracy={val_metrics['accuracy']:.3f}\n"
                        )
                except Exception as e:
                    print(f"Warning logging validation to file: {e}")

                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
                    torch.save(state_dict, checkpoint_path)
                    print(f"[SAVE] New best model: (f1={best_f1:.3f})")

                # Save last checkpoint for resume
                last_checkpoint_path = checkpoint_path.replace("best.pt", "last.pt")
                checkpoint_data = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_f1,
                }
                torch.save(checkpoint_data, last_checkpoint_path)

        # End of epoch validation
        print(f"\n[Epoch {epoch+1} Complete] Running validation...")
        val_metrics = run_epoch(model, val_loader, criterion, device)
        model.train()
        
        print(
            f"[Epoch {epoch+1} Complete] "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.3f} "
            f"val_precision={val_metrics['precision']:.3f} val_recall={val_metrics['recall']:.3f}"
        )

        # Log validation to file
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            val_log_path = os.path.join(os.path.dirname(checkpoint_path), "val_log.txt")
            with open(val_log_path, "a") as f:
                f.write(
                    f"[{timestamp}] Epoch {epoch+1} (End), Step {global_step}: "
                    f"loss={val_metrics['loss']:.4f}, f1={val_metrics['f1']:.3f}, "
                    f"precision={val_metrics['precision']:.3f}, recall={val_metrics['recall']:.3f}, "
                    f"accuracy={val_metrics['accuracy']:.3f}\n"
                )
        except Exception as e:
            print(f"Warning logging validation to file: {e}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save(state_dict, checkpoint_path)
            print(f"[SAVE] New best model: (f1={best_f1:.3f})")

        # Save last checkpoint at the end of epoch
        last_checkpoint_path = checkpoint_path.replace("best.pt", "last.pt")
        checkpoint_data = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
        }
        torch.save(checkpoint_data, last_checkpoint_path)

        # Save epoch-specific checkpoint
        epoch_checkpoint_path = checkpoint_path.replace("best.pt", f"epoch_{epoch + 1}.pt")
        torch.save(checkpoint_data, epoch_checkpoint_path)
        print(f"[SAVE] Saved epoch checkpoint to: {epoch_checkpoint_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-zarr",        required=True, type=str)
    parser.add_argument("--replay-legit-zarr", required=True, type=str)
    parser.add_argument("--replay-cheat-zarr", required=True, type=str)
    parser.add_argument("--beatmap-zarr",      required=True, type=str)
    parser.add_argument("--label-json",        required=True, type=str)
    parser.add_argument("--checkpoint-path",   required=True, type=str)

    args = parser.parse_args()

    train(
        pairs_folder=args.pairs_zarr,
        legit_folder=args.replay_legit_zarr,
        cheat_folder=args.replay_cheat_zarr,
        beatmaps_folder=args.beatmap_zarr,
        labels_path=args.label_json,
        checkpoint_path=args.checkpoint_path
    )


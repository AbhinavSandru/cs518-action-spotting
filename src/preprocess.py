"""
Preprocess SoccerNet features into fixed-size windows saved as .npz files.
Run once before training: python -m src.preprocess
This avoids loading full game files during training, making each batch read tiny.
"""
import os
import json
import numpy as np
from tqdm import tqdm
from SoccerNet.Downloader import getListGames

from src.dataset import LABELS, LABEL_TO_IDX

WINDOW_SIZE = 60
STRIDE = 30


def preprocess_split(data_dir, out_dir, split):
    game_list = getListGames(split=split)
    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)

    window_idx = 0
    for game_path in tqdm(game_list, desc=f"Preprocessing {split}"):
        game_dir = os.path.join(data_dir, game_path)
        feat1_path = os.path.join(game_dir, "1_ResNET_TF2.npy")
        feat2_path = os.path.join(game_dir, "2_ResNET_TF2.npy")
        label_path = os.path.join(game_dir, "Labels-v2.json")

        if not all(os.path.exists(p) for p in [feat1_path, feat2_path, label_path]):
            continue

        feat1 = np.load(feat1_path).astype(np.float32)
        feat2 = np.load(feat2_path).astype(np.float32)
        features = np.concatenate([feat1, feat2], axis=0)
        T = features.shape[0]
        half1_len = feat1.shape[0]

        # Build label array with Gaussian spreading over ±2 frames
        labels = np.zeros((T, len(LABELS)), dtype=np.float32)
        spread = 2  # frames either side (~1 second at 2fps)
        sigma = 1.0
        offsets = np.arange(-spread, spread + 1)
        weights = np.exp(-offsets**2 / (2 * sigma**2))
        weights /= weights.max()  # peak = 1.0

        with open(label_path) as f:
            annotation = json.load(f)
        for ann in annotation.get("annotations", []):
            half = int(ann["gameTime"].split(" - ")[0]) - 1
            time_str = ann["gameTime"].split(" - ")[1]
            minutes, seconds = map(int, time_str.split(":"))
            frame_idx = int((minutes * 60 + seconds) * 2)
            if half == 1:
                frame_idx += half1_len
            label = ann.get("label", "")
            if label not in LABEL_TO_IDX:
                continue
            cls_idx = LABEL_TO_IDX[label]
            for offset, weight in zip(offsets, weights):
                idx = frame_idx + offset
                if 0 <= idx < T:
                    labels[idx, cls_idx] = max(labels[idx, cls_idx], weight)

        # Save each window as a small .npz file
        for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            np.savez_compressed(
                os.path.join(split_out, f"{window_idx:07d}.npz"),
                features=features[start:end],
                labels=labels[start:end],
            )
            window_idx += 1

    print(f"  Saved {window_idx} windows for {split}")


if __name__ == "__main__":
    data_dir = "data/soccernet"
    out_dir = "data/windows_v2"

    for split in ["train", "valid", "test"]:
        preprocess_split(data_dir, out_dir, split)

    print("\nDone! Now update train.py to use WindowDataset.")

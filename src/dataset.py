import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from SoccerNet.Downloader import SoccerNetDownloader

LABELS = [
    "Ball out of play", "Throw-in", "Foul", "Indirect free-kick",
    "Clearance", "Shots on target", "Shots off target", "Corner",
    "Yellow card", "Goal", "Goalkeeper saves", "Direct free-kick",
    "Offside", "Substitution", "Yellow->red card", "Red card", "Kick-off"
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}


def download_features(data_dir, password):
    """Download precomputed ResNet-152 features from SoccerNet."""
    mng = SoccerNetDownloader(LocalDirectory=data_dir)
    mng.password = password
    mng.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy",
                              "Labels-v2.json"], split=["train", "valid", "test"])


class SoccerNetDataset(Dataset):
    def __init__(self, data_dir, split="train", window_size=60, stride=30):
        """
        Lazy-loading dataset — stores only window index pointers, reads from disk per batch.

        Args:
            data_dir:    path to soccernet data folder
            split:       one of "train", "valid", "test"
            window_size: number of frames per sliding window (at 2 fps)
            stride:      step size for sliding window
        """
        self.data_dir = data_dir
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.num_classes = len(LABELS)

        # Each entry: (feat1_path, feat2_path, label_path, start_frame, half1_len)
        self.index = []
        self._build_index()

    def _build_index(self):
        from SoccerNet.Downloader import getListGames
        game_list = getListGames(split=self.split)

        for game_path in game_list:
            game_dir = os.path.join(self.data_dir, game_path)
            feat1_path = os.path.join(game_dir, "1_ResNET_TF2.npy")
            feat2_path = os.path.join(game_dir, "2_ResNET_TF2.npy")
            label_path = os.path.join(game_dir, "Labels-v2.json")

            if not all(os.path.exists(p) for p in [feat1_path, feat2_path, label_path]):
                continue

            # Only read shape to build index — don't load data into RAM
            feat1 = np.load(feat1_path, mmap_mode="r")
            feat2 = np.load(feat2_path, mmap_mode="r")
            T1, T2 = feat1.shape[0], feat2.shape[0]
            T = T1 + T2

            for start in range(0, T - self.window_size + 1, self.stride):
                self.index.append((feat1_path, feat2_path, label_path, start, T1))

    def _load_labels(self, label_path, T, half1_len):
        labels = np.zeros((T, self.num_classes), dtype=np.float32)
        with open(label_path) as f:
            annotation = json.load(f)
        for ann in annotation.get("annotations", []):
            half = int(ann["gameTime"].split(" - ")[0]) - 1
            time_str = ann["gameTime"].split(" - ")[1]
            minutes, seconds = map(int, time_str.split(":"))
            frame_idx = int((minutes * 60 + seconds) * 2)
            if half == 1:
                frame_idx += half1_len
            if frame_idx < T:
                label = ann.get("label", "")
                if label in LABEL_TO_IDX:
                    labels[frame_idx, LABEL_TO_IDX[label]] = 1.0
        return labels

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        feat1_path, feat2_path, label_path, start, half1_len = self.index[idx]

        feat1 = np.load(feat1_path, mmap_mode="r").astype(np.float32)
        feat2 = np.load(feat2_path, mmap_mode="r").astype(np.float32)
        features = np.concatenate([feat1, feat2], axis=0)
        T = features.shape[0]

        end = start + self.window_size
        window_feat = features[start:end].copy()   # (window_size, 512)

        # Cache labels per game to avoid re-reading JSON for every window
        if not hasattr(self, '_label_cache'):
            self._label_cache = {}
        if label_path not in self._label_cache:
            self._label_cache[label_path] = self._load_labels(label_path, T, half1_len)
        labels = self._label_cache[label_path]
        window_labels = labels[start:end]          # (window_size, 17)

        return torch.tensor(window_feat), torch.tensor(window_labels)


class WindowDataset(Dataset):
    """Fast dataset that reads preprocessed .npz window files from disk."""

    def __init__(self, windows_dir, split):
        self.split_dir = os.path.join(windows_dir, split)
        self.files = sorted(f for f in os.listdir(self.split_dir) if f.endswith(".npz") and not f.startswith("._"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.split_dir, self.files[idx]))
        return torch.tensor(data["features"]), torch.tensor(data["labels"])

"""
Find the best threshold per class on the validation set.
Run: python -m src.tune_thresholds
"""
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from SoccerNet.Downloader import getListGames

from src.model import ActionSpotter
from src.dataset import LABELS
from src.evaluate import predict_game, nms
from SoccerNet.Evaluation.ActionSpotting import evaluate as soccernet_evaluate


def tune_thresholds(
    data_dir="data/soccernet",
    checkpoint_path="checkpoints/best_model.pt",
    output_dir="results/predictions_val",
    window_size=60,
    stride=30,
    nms_window=10,
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        model = ActionSpotter(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = ActionSpotter(input_dim=2048).to(device)
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # Collect raw probs per game on validation set
    game_list = getListGames(split="valid")
    all_probs = []   # list of (probs, half1_len, game_path)

    for game_path in tqdm(game_list, desc="Running inference on val"):
        game_dir = os.path.join(data_dir, game_path)
        feat1 = os.path.join(game_dir, "1_ResNET_TF2.npy")
        feat2 = os.path.join(game_dir, "2_ResNET_TF2.npy")
        if not (os.path.exists(feat1) and os.path.exists(feat2)):
            continue
        probs, half1_len = predict_game(model, feat1, feat2, window_size, stride, device)
        all_probs.append((probs, half1_len, game_path))

    # Try different thresholds and pick best per class
    thresholds = np.arange(0.05, 0.55, 0.05)
    best_thresholds = [0.2] * len(LABELS)
    best_maps = [0.0] * len(LABELS)

    for thresh in thresholds:
        os.makedirs(output_dir, exist_ok=True)
        for probs, half1_len, game_path in all_probs:
            preds = nms(probs, half1_len, threshold=float(thresh), nms_window=nms_window)
            out_path = os.path.join(output_dir, game_path, "results_spotting.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"predictions": preds}, f, indent=2)

        results = soccernet_evaluate(
            SoccerNet_path=data_dir,
            Predictions_path=output_dir,
            split="valid",
            version=2,
            metric="tight",
        )
        per_class = results["a_mAP_per_class"]
        for i, m in enumerate(per_class):
            if float(m) > best_maps[i]:
                best_maps[i] = float(m)
                best_thresholds[i] = float(thresh)

    print("\nBest thresholds per class:")
    print(f"{'Class':<25} {'Threshold':>10} {'Val mAP@1s':>12}")
    print("-" * 50)
    for i, label in enumerate(LABELS):
        print(f"  {label:<23} {best_thresholds[i]:>10.2f} {best_maps[i]:>12.4f}")

    # Save thresholds
    out = {"thresholds": best_thresholds}
    with open("checkpoints/best_thresholds.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to checkpoints/best_thresholds.json")
    return best_thresholds


if __name__ == "__main__":
    tune_thresholds()

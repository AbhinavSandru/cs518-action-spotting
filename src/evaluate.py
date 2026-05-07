import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from SoccerNet.Evaluation.ActionSpotting import evaluate as soccernet_evaluate

from src.dataset import SoccerNetDataset, LABELS
from src.model import ActionSpotter


def predict_game(model, feat1_path, feat2_path, window_size, stride, device):
    """
    Run inference on a single game and return per-frame probability scores.

    Returns:
        probs: numpy array of shape (T, num_classes)
    """
    feat1 = np.load(feat1_path).astype(np.float32)
    feat2 = np.load(feat2_path).astype(np.float32)
    features = np.concatenate([feat1, feat2], axis=0)  # (T, 512)
    T = features.shape[0]

    probs = np.zeros((T, len(LABELS)), dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window = torch.tensor(features[start:end]).unsqueeze(0).to(device)  # (1, T, 512)
            logits = model(window)                                               # (1, T, 17)
            window_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()       # (T, 17)
            probs[start:end] += window_probs
            counts[start:end] += 1

    # Average overlapping windows
    counts = np.maximum(counts, 1)
    probs = probs / counts[:, None]

    return probs, feat1.shape[0]  # also return half1 length for timestamp calc


def nms(probs, half1_len, threshold=0.5, nms_window=10):
    """
    Apply per-class non-maximum suppression to get final predictions.

    Args:
        probs:      (T, num_classes) probability array
        half1_len:  number of frames in first half (for half assignment)
        threshold:  confidence threshold
        nms_window: suppress predictions within this many frames of a peak

    Returns:
        predictions: list of dicts with keys 'half', 'position', 'label', 'confidence'
    """
    predictions = []
    T, num_classes = probs.shape

    # threshold can be a float (global) or a list (per-class)
    thresholds = threshold if isinstance(threshold, list) else [threshold] * num_classes

    for cls in range(num_classes):
        cls_probs = probs[:, cls].copy()
        while True:
            peak_idx = np.argmax(cls_probs)
            peak_val = cls_probs[peak_idx]
            if peak_val < thresholds[cls]:
                break

            # Determine half and position in seconds
            if peak_idx < half1_len:
                half = 1
                position_sec = peak_idx / 2  # 2 fps
            else:
                half = 2
                position_sec = (peak_idx - half1_len) / 2

            predictions.append({
                "half": half,
                "position": int(position_sec * 1000),  # milliseconds for SoccerNet eval
                "label": LABELS[cls],
                "confidence": float(peak_val),
            })

            # Suppress nearby frames
            lo = max(0, peak_idx - nms_window)
            hi = min(T, peak_idx + nms_window)
            cls_probs[lo:hi] = 0

    return predictions


def run_evaluation(
    data_dir="data/soccernet",
    checkpoint_path="checkpoints/best_model.pt",
    split="test",
    window_size=60,
    stride=30,
    threshold=0.5,
    nms_window=10,
    output_dir="results/predictions",
):
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model — supports both old (state_dict only) and new (dict with config) checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        model = ActionSpotter(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = ActionSpotter().to(device)
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load per-class thresholds if available
    threshold_path = "checkpoints/best_thresholds.json"
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = json.load(f)["thresholds"]
        print(f"Using per-class thresholds from {threshold_path}")
    else:
        print(f"Using global threshold: {threshold}")

    from SoccerNet.Downloader import getListGames
    game_list = getListGames(split=split)

    # Run inference per game and save predictions
    for game_path in tqdm(game_list, desc=f"Evaluating {split}"):
        game_dir = os.path.join(data_dir, game_path)
        feat1 = os.path.join(game_dir, "1_ResNET_TF2.npy")
        feat2 = os.path.join(game_dir, "2_ResNET_TF2.npy")
        if not (os.path.exists(feat1) and os.path.exists(feat2)):
            continue

        probs, half1_len = predict_game(model, feat1, feat2, window_size, stride, device)
        preds = nms(probs, half1_len, threshold=threshold, nms_window=nms_window)

        # Save predictions in SoccerNet format
        out_path = os.path.join(output_dir, game_path, "results_spotting.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"predictions": preds}, f, indent=2)

    # Run official SoccerNet evaluation at each tolerance
    print("\nRunning official SoccerNet evaluation...")

    results_tight = soccernet_evaluate(
        SoccerNet_path=data_dir,
        Predictions_path=output_dir,
        split=split,
        version=2,
        metric="tight",
    )
    results_loose = soccernet_evaluate(
        SoccerNet_path=data_dir,
        Predictions_path=output_dir,
        split=split,
        version=2,
        metric="loose",
    )

    from src.dataset import LABELS
    tight_per_class = results_tight["a_mAP_per_class"]
    loose_per_class = results_loose["a_mAP_per_class"]

    print(f"\n{'='*55}")
    print(f"{'Class':<25} {'mAP@1s':>8} {'mAP@5s':>8}")
    print(f"{'-'*55}")
    for i, label in enumerate(LABELS):
        print(f"  {label:<23} {float(tight_per_class[i]):>8.4f} {float(loose_per_class[i]):>8.4f}")
    print(f"{'-'*55}")
    print(f"  {'Average-mAP':<23} {results_tight['a_mAP']:>8.4f} {results_loose['a_mAP']:>8.4f}")
    print(f"{'='*55}")

    return results_tight


if __name__ == "__main__":
    run_evaluation()

"""
Option 2 — Download one game's video and extract clips around top predictions.
Step 1: python3 demo_video.py --download      (downloads ~4GB video)
Step 2: python3 demo_video.py --clips         (extracts clips + overlays)
Step 3: python3 demo_video.py --montage       (stitches clips into one demo video)
"""
import os
import json
import argparse
import subprocess
from collections import defaultdict

GAME_PATH = "england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester"
DATA_DIR  = "data/soccernet"
PRED_FILE = f"results/predictions/{GAME_PATH}/results_spotting.json"
CLIPS_DIR = "results/demo_clips"
VIDEO_DIR = os.path.join(DATA_DIR, GAME_PATH)

# Classes we care about for the demo (skip boring ones)
SHOWCASE_CLASSES = [
    "Goal", "Shots on target", "Corner", "Yellow card",
    "Foul", "Substitution", "Offside", "Direct free-kick",
]

# ── helpers ───────────────────────────────────────────────────────────────────
def ms_to_sec(ms):
    return ms / 1000.0

def format_time(ms):
    mins = ms // 60000
    secs = (ms % 60000) // 1000
    return f"{int(mins):02d}:{int(secs):02d}"


# ── Step 1: Download video ─────────────────────────────────────────────────────
def download_video():
    print("Downloading video for:", GAME_PATH)
    print("This will take a few minutes (~4 GB)...\n")
    from SoccerNet.Downloader import SoccerNetDownloader
    mng = SoccerNetDownloader(LocalDirectory=DATA_DIR)
    mng.password = "s0cc3rn3t"
    mng.downloadGames(
        files=["1_720p.mkv", "2_720p.mkv"],
        split=["test"],
        task="clips",
    )
    # Check if downloaded
    for half in [1, 2]:
        p = os.path.join(VIDEO_DIR, f"{half}_720p.mkv")
        if os.path.exists(p):
            print(f"  ✓ Half {half}: {p}")
        else:
            print(f"  ✗ Half {half} not found — trying lower quality...")
            mng.downloadGames(files=[f"{half}_224p.mkv"], split=["test"])


# ── Step 2: Extract clips ──────────────────────────────────────────────────────
def extract_clips():
    import cv2
    import numpy as np
    from tqdm import tqdm

    os.makedirs(CLIPS_DIR, exist_ok=True)

    with open(PRED_FILE) as f:
        preds = json.load(f)["predictions"]

    # Pick top prediction per showcase class, highest confidence
    best = {}
    for p in preds:
        cls = p["label"]
        if cls not in SHOWCASE_CLASSES:
            continue
        if cls not in best or p["confidence"] > best[cls]["confidence"]:
            best[cls] = p

    print(f"\nExtracting {len(best)} clips...\n")

    clips_made = []
    for cls, p in best.items():
        half     = p["half"]
        pos_ms   = p["position"]
        conf     = p["confidence"]
        time_str = format_time(pos_ms)

        # Find video file
        video_file = None
        for ext in ["720p.mkv", "224p.mkv", "720p.mp4", "224p.mp4"]:
            candidate = os.path.join(VIDEO_DIR, f"{half}_{ext}")
            if os.path.exists(candidate):
                video_file = candidate
                break

        if not video_file:
            print(f"  ✗ No video for half {half} — skipping {cls}")
            continue

        start_sec = max(0, ms_to_sec(pos_ms) - 8)
        duration  = 16
        safe_cls  = cls.replace(" ", "_").replace("->", "_to_")
        out_path  = os.path.join(CLIPS_DIR, f"{safe_cls}_H{half}_{time_str.replace(':', 'm')}s.mp4")

        print(f"  {cls:<22}  Half {half}  {time_str}  conf={conf:.2f}")

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        total_frames = int(duration * fps)
        for _ in tqdm(range(total_frames), desc=f"    writing", leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            # Overlay: black semi-transparent bar at top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Class label (large, coloured)
            cv2.putText(frame, cls, (20, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (100, 220, 140), 2, cv2.LINE_AA)
            # Details (smaller, white)
            detail = f"Half {half}   {time_str}   confidence: {conf:.2f}"
            cv2.putText(frame, detail, (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

            out.write(frame)

        cap.release()
        out.release()
        print(f"    → {out_path}")
        clips_made.append((cls, out_path, half, time_str, conf))

    with open(os.path.join(CLIPS_DIR, "clip_list.json"), "w") as f:
        json.dump(clips_made, f, indent=2)

    print(f"\nDone — {len(clips_made)} clips saved to {CLIPS_DIR}/")
    return clips_made


# ── Step 3: Stitch into montage ────────────────────────────────────────────────
def make_montage():
    list_file = os.path.join(CLIPS_DIR, "clip_list.json")
    if not os.path.exists(list_file):
        print("Run --clips first.")
        return

    with open(list_file) as f:
        clips = json.load(f)

    if not clips:
        print("No clips found.")
        return

    # Write ffmpeg concat list
    concat_path = os.path.join(CLIPS_DIR, "concat.txt")
    with open(concat_path, "w") as f:
        for _, path, *_ in clips:
            f.write(f"file '{os.path.abspath(path)}'\n")

    out_path = "results/demo_montage.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac",
        out_path
    ]
    print("Stitching montage...")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        print(f"Montage saved → {out_path}")
    else:
        print("ffmpeg error:", result.stderr.decode()[:300])


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download video")
    parser.add_argument("--clips",    action="store_true", help="Extract clips")
    parser.add_argument("--montage",  action="store_true", help="Stitch montage")
    parser.add_argument("--all",      action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.all or args.download:
        download_video()
    if args.all or args.clips:
        extract_clips()
    if args.all or args.montage:
        make_montage()

    if not any([args.download, args.clips, args.montage, args.all]):
        print("Usage:")
        print("  python3 demo_video.py --download   # download ~4GB video")
        print("  python3 demo_video.py --clips      # extract action clips")
        print("  python3 demo_video.py --montage    # stitch into one video")
        print("  python3 demo_video.py --all        # run everything")

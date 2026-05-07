"""
Downloads ONLY Man Utd 4-1 Leicester video (both halves) with live progress.
Run: python3 download_game.py
"""
import os
import sys
import time
import threading
from SoccerNet.Downloader import SoccerNetDownloader

GAME    = "england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester"
OUT_DIR = f"data/soccernet/{GAME}"
os.makedirs(OUT_DIR, exist_ok=True)

mng          = SoccerNetDownloader(LocalDirectory="data/soccernet")
mng.password = "s0cc3rn3t"

print("=" * 60)
print("  Manchester United 4 – 1 Leicester City  |  Sep 24 2016")
print("=" * 60)

def live_progress(filepath, total_mb, label, stop_event):
    """Prints a live progress bar by watching the file size."""
    bar_width = 30
    while not stop_event.is_set():
        if os.path.exists(filepath):
            done_mb = os.path.getsize(filepath) / 1e6
            pct     = min(done_mb / total_mb, 1.0)
            filled  = int(bar_width * pct)
            bar     = "█" * filled + "░" * (bar_width - filled)
            sys.stdout.write(f"\r  {label}  [{bar}]  {done_mb:.0f}/{total_mb:.0f} MB  {pct*100:.1f}%")
            sys.stdout.flush()
            if pct >= 1.0:
                break
        time.sleep(1)
    sys.stdout.write("\n")

# Approx sizes for 720p halves (~800 MB each)
SIZES = {1: 800, 2: 800}

for half in [1, 2]:
    fname = f"{half}_720p.mkv"
    dest  = os.path.join(OUT_DIR, fname)

    if os.path.exists(dest) and os.path.getsize(dest) > 50_000_000:
        print(f"\nHalf {half}: already downloaded ({os.path.getsize(dest)/1e6:.0f} MB) ✓")
        continue

    print(f"\nHalf {half} / 2  —  downloading {fname}...")

    stop_evt = threading.Event()
    t = threading.Thread(
        target=live_progress,
        args=(dest, SIZES[half], f"Half {half}", stop_evt),
        daemon=True
    )
    t.start()

    try:
        mng.downloadGame(game=GAME, files=[fname], spl="test", verbose=False)
    except Exception as e:
        stop_evt.set()
        t.join()
        print(f"\n  720p failed ({e}), trying 224p...")
        fname = f"{half}_224p.mkv"
        dest  = os.path.join(OUT_DIR, fname)
        stop_evt2 = threading.Event()
        t2 = threading.Thread(
            target=live_progress,
            args=(dest, 200, f"Half {half} (224p)", stop_evt2),
            daemon=True
        )
        t2.start()
        mng.downloadGame(game=GAME, files=[fname], spl="test", verbose=False)
        stop_evt2.set(); t2.join()
    else:
        stop_evt.set()
        t.join()

    if os.path.exists(dest):
        print(f"  ✓  Saved: {dest}  ({os.path.getsize(dest)/1e6:.0f} MB)")
    else:
        print(f"  ✗  Download failed for half {half}")

print("\n" + "=" * 60)
print("  Done! Now run:  python3 demo_video.py --clips")
print("=" * 60)

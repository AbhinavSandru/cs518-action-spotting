# Action Spotting in Soccer Broadcast Videos

**CS518: Deep Learning for Computer Vision**  
Abhinav Sandru · Gaurav Chintakunta · Raviteja Ravella

---

## Overview

This project tackles **action spotting** on the [SoccerNet-v2](https://soccer-net.org/) benchmark — the task of identifying the precise timestamps of 17 predefined action classes (goals, fouls, cards, corners, substitutions, etc.) in full 90-minute broadcast soccer matches.

We train a lightweight **Transformer encoder** on precomputed ResNet-152 appearance features, using focal loss and sqrt inverse-frequency class weighting to handle extreme temporal sparsity (~1 positive frame per 1000 per class).

**Results on SoccerNet-v2 test set: 38.18% mAP@1s · 52.86% mAP@5s**

---

## Repository Structure

```
cs518-action-spotting/
├── src/
│   ├── model.py            # ActionSpotter Transformer encoder
│   ├── train.py            # Training loop with focal loss
│   ├── evaluate.py         # Sliding-window inference + NMS
│   ├── preprocess.py       # Feature windowing + Gaussian label spreading
│   ├── dataset.py          # WindowDataset for .npz files
│   └── tune_thresholds.py  # Per-class threshold tuning on validation set
├── results/
│   ├── loss_curve.png
│   └── timeline_demo.png
├── CS518_ActionSpotting_Demo.ipynb           # Demo notebook (synthetic data)
├── CS518_ActionSpotting_Demo_executed.ipynb  # Pre-executed version
├── main.py                 # Entry point (train / evaluate)
├── demo_timeline.py        # Visualize predictions on a real game
├── demo_video.py           # Extract action clips from match video
├── download_game.py        # Download a single SoccerNet game video
├── requirements.txt
└── report_abhinav.pdf / report_gaurav.pdf / report_raviteja.pdf
```

---

## Model Architecture

The **ActionSpotter** model performs per-frame binary classification across all 17 action classes simultaneously:

1. **L2 Normalization** — removes scale variance across games and cameras
2. **Linear Projection** — 2048 → 256 dimensions
3. **Positional Embedding** — learned, shape (1, 1000, 256)
4. **Transformer Encoder** — 4 layers, 4 attention heads, d_ff=1024, dropout=0.3
5. **Classification Head** — Linear(256 → 17) per frame

Output: `(B, T, 17)` logits. Timestamps are extracted post-hoc via NMS on per-frame score arrays.  
Total parameters: ~1.4 million.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Demo (no SoccerNet required)

Open `CS518_ActionSpotting_Demo.ipynb` — it generates synthetic data and runs the full pipeline end-to-end in a few minutes.

### 3. Full pipeline (requires SoccerNet access)

**Preprocess** (run once — generates ~200K .npz windows):
```bash
python -m src.preprocess
```

**Train:**
```bash
python main.py --mode train
```

**Evaluate on test set:**
```bash
python main.py --mode evaluate
```

**Tune per-class thresholds on validation set:**
```bash
python -m src.tune_thresholds
```

---

## Results

| Configuration | mAP@1s | mAP@5s |
|---|---|---|
| Baseline (plain BCE) | ~0% | ~0% |
| + Sqrt class weights | ~24% | ~39% |
| + Focal loss (γ=2) | ~28% | ~44% |
| + L2 normalization | 31.94% | ~48% |
| + Per-class thresholds | **38.18%** | **52.86%** |

Top performing classes: Substitution (69.5% mAP@1s), Foul (62.8%), Yellow card (55.7%).

---

## Hardware

Trained on Apple MacBook Pro M4 Pro (24GB) using PyTorch MPS backend.  
Convergence in ~15–20 epochs (~3 hours). SoccerNet features stored on external SSD.

---

## Reports

Individual project reports (same main body, individually written appendix):
- `report_abhinav.pdf`
- `report_gaurav.pdf`
- `report_raviteja.pdf`

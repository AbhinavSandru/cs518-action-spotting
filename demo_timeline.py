"""
Option 1 — Interactive timeline visualization of model predictions for one game.
Run: python3 demo_timeline.py
Shows a matplotlib plot of every detected action across 90 minutes.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

GAME      = "england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester"
PRED_FILE = f"results/predictions/{GAME}/results_spotting.json"
GAME_LABEL = "Manchester United 4 – 1 Leicester City  |  24 Sep 2016"

# ── colour per class ──────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Goal":               "#6ED68A",
    "Foul":               "#F5C542",
    "Yellow card":        "#FFD700",
    "Red card":           "#FF4444",
    "Yellow->red card":   "#FF8C00",
    "Corner":             "#4FB3D9",
    "Shots on target":    "#89DCEB",
    "Shots off target":   "#A8D8EA",
    "Substitution":       "#C792EA",
    "Kick-off":           "#AAAAAA",
    "Throw-in":           "#888888",
    "Indirect free-kick": "#FFA07A",
    "Direct free-kick":   "#FF7F50",
    "Clearance":          "#66BB6A",
    "Offside":            "#EF9A9A",
    "Goalkeeper saves":   "#80DEEA",
    "Ball out of play":   "#555555",
}

# ── load predictions ──────────────────────────────────────────────────────────
with open(PRED_FILE) as f:
    preds = json.load(f)["predictions"]

# Filter to high-confidence only for cleanliness
MIN_CONF = 0.4
preds = [p for p in preds if p["confidence"] >= MIN_CONF]

# Sort classes by how interesting they are for display
CLASS_ORDER = [
    "Goal", "Shots on target", "Shots off target", "Corner",
    "Foul", "Yellow card", "Red card", "Yellow->red card",
    "Substitution", "Offside", "Direct free-kick", "Indirect free-kick",
    "Clearance", "Goalkeeper saves", "Throw-in", "Kick-off", "Ball out of play"
]

# Only keep classes that actually have predictions
present = set(p["label"] for p in preds)
classes = [c for c in CLASS_ORDER if c in present]
class_y = {c: i for i, c in enumerate(classes)}

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
fig.patch.set_facecolor("#0F0F1A")
fig.suptitle(f"Model Predictions\n{GAME_LABEL}", color="white",
             fontsize=16, fontweight="bold", y=0.98)

for ax_idx, half in enumerate([1, 2]):
    ax = axes[ax_idx]
    ax.set_facecolor("#1A1A2E")
    ax.set_title(f"Half {half}", color="#4FB3D9", fontsize=14, fontweight="bold", pad=8)

    half_preds = [p for p in preds if p["half"] == half]

    for p in half_preds:
        mins = p["position"] / 60000
        y    = class_y[p["label"]]
        conf = p["confidence"]
        color = CLASS_COLORS.get(p["label"], "#FFFFFF")
        size  = 40 + conf * 120   # bigger = more confident

        ax.scatter(mins, y, s=size, color=color, alpha=0.85,
                   edgecolors="white", linewidths=0.3, zorder=3)

    # Vertical grid lines every 15 minutes
    for m in range(0, 56, 15):
        ax.axvline(m, color="#333355", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(m + 0.3, len(classes) - 0.3, f"{m}'",
                color="#666688", fontsize=8)

    ax.set_xlim(-1, 50)
    ax.set_ylim(-0.8, len(classes) - 0.2)
    ax.set_xlabel("Match Minute", color="#AAAAAA", fontsize=11)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, color="white", fontsize=10)
    ax.tick_params(colors="#AAAAAA", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

axes[0].set_ylabel("Action Class", color="#AAAAAA", fontsize=11)

# Legend: size = confidence
for conf, lbl in [(0.4, "conf 0.4"), (0.6, "conf 0.6"), (0.8, "conf 0.8")]:
    size = 40 + conf * 120
    axes[1].scatter([], [], s=size, color="white", alpha=0.8, label=lbl)
axes[1].legend(title="Dot size", title_fontsize=9, fontsize=9,
               facecolor="#1A1A2E", edgecolor="#4FB3D9",
               labelcolor="white", loc="lower right")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "results/timeline_demo.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F0F1A")
print(f"Saved → {out}")
plt.show()

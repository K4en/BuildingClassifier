import os
import json
from pathlib import Path

# Config
VAL_LOG_PATH = Path("checkpoints/best_val_predictions.json")
REPLAY_DIR = Path("replay")
REINFORCE_PATH = REPLAY_DIR / "reinforce.json"
FOCUS_PATH = REPLAY_DIR / "focus.json"

# Confidence thresholds
REINFORCE_THRESHOLD = 0.95
FOCUS_THRESHOLD = 0.65

def extract_pools():
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    with open(VAL_LOG_PATH, "r") as f:
        predictions = json.load(f)

    reinforce = []
    focus = []

    for p in predictions:
        if p["correct"] and p["confidence"] >= REINFORCE_THRESHOLD:
            reinforce.append(p)
        elif not p["correct"] and p["confidence"] <= REINFORCE_THRESHOLD:
            focus.append(p)

    print(f" Reinforce pool: {len(reinforce)} samples")
    print(f" Focus pool: {len(focus)} samples")

    with open(REINFORCE_PATH, "w") as f:
        json.dump(reinforce, f, indent=2)

    with open(FOCUS_PATH, "w") as f:
        json.dump(focus, f, indent=2)

    print("Replay pools saved to 'replay/'")

if __name__ == "__main__":
    extract_pools()

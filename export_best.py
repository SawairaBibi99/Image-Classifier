# export_best.py
import os
from tensorflow import keras
from train import build_model  # uses the same architecture as training

BEST = "best.keras"
OUT_DIR = "saved_model"
OUT_PATH = os.path.join(OUT_DIR, "classifier.keras")

assert os.path.exists(BEST), "best.keras not found (did training run at least one epoch?)"
os.makedirs(OUT_DIR, exist_ok=True)

model, _ = build_model()         # build same model
model.load_weights(BEST)         # load checkpoint weights
model.save(OUT_PATH)             # save final model file
print(f"[INFO] Exported to {OUT_PATH}")

# ==============================================================================
# main.py
# ------------------------------------------------------------------------------
# Entry point for the Wine Quality ANN pipeline.
#
# This script runs the full pipeline end-to-end:
#   1. Load & preprocess raw data
#   2. Build the ANN model
#   3. Train the model
#   4. Evaluate and save results
#
# Usage:
#   python main.py
#
# Note:
#   The notebooks in notebooks/ are the primary way to explore each step
#   interactively. This file ties everything together for a single clean run.
# ==============================================================================

import os
import sys

# ── Make sure src/ is importable ──────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))

from src.data_preprocessing import load_data, preprocess
from src.model import build_model, load_best_params
from src.train import train_model
from src.evaluate import evaluate_model

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DATA    = "data/raw/WineQT.csv"
MODEL_DIR   = "models/"
PARAMS_FILE = "models/best_params.json"

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    """Run the full Wine Quality ANN pipeline using tuned hyperparameters."""

    print("=" * 60)
    print("  🍷 Wine Quality Prediction — ANN Pipeline")
    print("=" * 60)

    # ── Step 0: Load tuned hyperparameters ─────────────────────────────────────
    # This pipeline is documented (README, presentation) as producing "the
    # optimized model" from Step 4's grid search. If best_params.json is
    # missing, we stop here rather than silently falling back to generic
    # defaults and producing an undocumented, unrelated model.
    print("\n⚙️  Step 0: Loading tuned hyperparameters...")
    try:
        best_params = load_best_params(PARAMS_FILE)
    except (FileNotFoundError, KeyError) as e:
        print(f"\n❌ {e}")
        print("   main.py requires Step 4's grid search results to run "
              "the documented optimized pipeline. Aborting.")
        sys.exit(1)

    # ── Step 1: Load & preprocess ──────────────────────────────────────────────
    print("\n📂 Step 1: Loading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(RAW_DATA)
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
    print(f"   Train : {X_train.shape[0]:,} samples")
    print(f"   Val   : {X_val.shape[0]:,} samples")
    print(f"   Test  : {X_test.shape[0]:,} samples")

    # ── Step 2: Build model ────────────────────────────────────────────────────
    print("\n🧠 Step 2: Building ANN model (tuned hyperparameters)...")
    model = build_model(
        input_dim=X_train.shape[1],
        units=best_params["units"],
        learning_rate=best_params["learning_rate"],
        l2_lambda=best_params["l2_lambda"],
        dropout_rate=best_params["dropout_rate"],
    )
    model.summary()

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    print("\n🏋️  Step 3: Training model...")
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=best_params["batch_size"],
    )

    # ── Step 4: Evaluate ───────────────────────────────────────────────────────
    print("\n📊 Step 4: Evaluating model...")
    evaluate_model(model, X_test, y_test, history)

    print("\n✅ Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
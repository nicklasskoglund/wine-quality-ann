# ==============================================================================
# src/model.py
# ------------------------------------------------------------------------------
# Defines ANN architectures for the Wine Quality binary classification task.
#
# Functions:
#   build_baseline_model(input_dim)  → simple, unregularized baseline ANN
#   build_model(input_dim, ...)      → regularized ANN (L2 + Dropout),
#                                        used for Grid Search / optimization (Step 4)
#
# --- Baseline architecture (build_baseline_model) ---
#   Input layer    → input_dim features (11)
#   Hidden layer 1 → 16 neurons, ReLU
#   Hidden layer 2 → 8 neurons, ReLU
#   Output layer   → 1 neuron, Sigmoid (binary classification)
#
# --- Optimized architecture (build_model) ---
#   Input layer    → input_dim features (11)
#   Hidden layer 1 → `units` neurons, ReLU, L2 regularization, Dropout
#   Hidden layer 2 → `units` // 2 neurons, ReLU, L2 regularization, Dropout
#   Output layer   → 1 neuron, Sigmoid (binary classification)
#
#   Actual hyperparameter values (units, dropout_rate, l2_lambda,
#   learning_rate, batch_size) come from Step 4's grid search results —
#   see load_best_params() below and models/best_params.json. The module
#   constants further down are generic fallback defaults only, used when
#   no tuned parameters are supplied; they do not represent "the
#   optimized model".
#
# Loss function : Binary Crossentropy
# Optimizer     : Adam
# Metric        : Accuracy (baseline) / Accuracy, AUC (optimized)
# ==============================================================================

import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ── Constants ──────────────────────────────────────────────────────────────────
UNITS         = 64      # Neurons in build_model's hidden layer 1 (hidden layer 2 uses UNITS // 2)
LEARNING_RATE = 0.001   # Adam optimizer default learning rate
L2_LAMBDA     = 0.001   # L2 regularization strength (used in build_model)
DROPOUT_RATE  = 0.3     # Fraction of neurons dropped during training (used in build_model)

# Baseline architecture sizes
BASELINE_HIDDEN_1_UNITS = 16   # Neurons in baseline hidden layer 1
BASELINE_HIDDEN_2_UNITS = 8    # Neurons in baseline hidden layer 2


def build_baseline_model(input_dim: int) -> keras.Sequential:
    """
    Build and compile a simple baseline ANN for binary wine quality classification.

    This architecture is intentionally simple — two small hidden layers with
    ReLU activation, no regularization — to serve as a reference point
    ("baseline") against which the regularized, tuned architecture
    (see build_model) can be compared in Step 4 (Grid Search optimization).

    Architecture:
        Input → Dense(16, ReLU) → Dense(8, ReLU) → Dense(1, Sigmoid)

    Args:
        input_dim (int): Number of input features (e.g. 11 for WineQT.csv after preprocessing)

    Returns:
        keras.Sequential: Compiled Keras model, ready for training
    """
    # ── Build model ────────────────────────────────────────────────────────────
    model = keras.Sequential(
        [
            # ── Input ──────────────────────────────────────────────────────────
            keras.Input(shape=(input_dim,), name="input"),

            # ── Hidden layer 1 ─────────────────────────────────────────────────
            layers.Dense(BASELINE_HIDDEN_1_UNITS, activation="relu", name="hidden_1"),

            # ── Hidden layer 2 ─────────────────────────────────────────────────
            layers.Dense(BASELINE_HIDDEN_2_UNITS, activation="relu", name="hidden_2"),

            # ── Output layer ───────────────────────────────────────────────────
            layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="baseline_ann",
    )

    # ── Compile ────────────────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_model(
    input_dim: int,
    units: int            = UNITS,
    learning_rate: float = LEARNING_RATE,
    l2_lambda: float     = L2_LAMBDA,
    dropout_rate: float  = DROPOUT_RATE,
) -> keras.Sequential:
    """
    Build and compile the regularized ANN model for binary wine quality
    classification. Used for Grid Search optimization (Step 4).

    The model uses:
        - ReLU activation in hidden layers (avoids vanishing gradient)
        - L2 regularization (penalizes large weights → reduces overfitting)
        - Dropout (randomly disables neurons during training → reduces overfitting)
        - Sigmoid output (squashes output to [0, 1] → interpreted as probability)
        - Binary Crossentropy loss (standard for binary classification)
        - Adam optimizer (adaptive learning rate → fast and stable convergence)

    Args:
        input_dim     (int)   : Number of input features (columns in X_train)
        units         (int)   : Number of neurons in hidden layer 1.
                                 Hidden layer 2 uses units // 2 neurons.
        learning_rate (float) : Learning rate for Adam optimizer
        l2_lambda     (float) : L2 regularization penalty strength
        dropout_rate  (float) : Fraction of neurons to drop during training

    Returns:
        keras.Sequential: Compiled Keras model ready for training
    """
    # ── Build model ────────────────────────────────────────────────────────────
    model = keras.Sequential(
        [
            # ── Input ──────────────────────────────────────────────────────────
            keras.Input(shape=(input_dim,), name="input"),

            # ── Hidden layer 1 ─────────────────────────────────────────────────
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.L2(l2_lambda),
                name="hidden_1",
            ),
            layers.Dropout(dropout_rate, name="dropout_1"),

            # ── Hidden layer 2 ─────────────────────────────────────────────────
            layers.Dense(
                units // 2,
                activation="relu",
                kernel_regularizer=regularizers.L2(l2_lambda),
                name="hidden_2",
            ),
            layers.Dropout(dropout_rate, name="dropout_2"),

            # ── Output layer ───────────────────────────────────────────────────
            layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="wine_quality_ann",
    )

    # ── Compile ────────────────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def load_best_params(filepath: str = "models/best_params.json") -> dict:
    """
    Load tuned hyperparameters produced by the Step 4 grid search.

    This is the single source of truth for the winning hyperparameters
    (units, dropout_rate, l2_lambda, learning_rate, batch_size). Reading
    from this file — rather than hardcoding the winning values as module
    constants — prevents the code and the grid search results from
    silently drifting apart if the grid search is ever re-run with
    different results.

    By design, this function does NOT fall back to guessed defaults if
    the file is missing: main.py's pipeline is documented (in the README
    and presentation) as producing "the optimized model", so running it
    without the actual winning hyperparameters would silently produce a
    different, undocumented model. Callers are expected to stop the run
    and surface a clear error instead.

    Args:
        filepath (str): Path to the JSON file saved by the grid search
                         notebook (03_optimization.ipynb)

    Returns:
        dict: Hyperparameters with keys:
              "units", "dropout_rate", "l2_lambda",
              "learning_rate", "batch_size"
              (plus result fields "val_auc", "test_auc", "test_accuracy"
              kept for traceability, though not used by build_model/
              train_model)

    Raises:
        FileNotFoundError: If `filepath` does not exist — typically means
                            Step 4 (grid search) has not been run yet.
        KeyError: If the file exists but is missing one or more of the
                   required hyperparameter keys.
    """
    # ── Check file exists ──────────────────────────────────────────────────────
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find tuned hyperparameters at '{filepath}'. "
            f"This file is produced by the Step 4 grid search "
            f"(notebooks/03_optimization.ipynb). Run that notebook first "
            f"to generate it before running main.py."
        )

    # ── Load and validate ──────────────────────────────────────────────────────
    with open(filepath, "r") as f:
        params = json.load(f)

    required_keys = {"units", "dropout_rate", "l2_lambda", "learning_rate", "batch_size"}
    missing_keys = required_keys - params.keys()
    if missing_keys:
        raise KeyError(
            f"'{filepath}' is missing required hyperparameter key(s): "
            f"{sorted(missing_keys)}. Expected keys: {sorted(required_keys)}."
        )

    print(f"   Loaded tuned hyperparameters ← {filepath}")
    print(f"   units={params['units']}, dropout_rate={params['dropout_rate']}, "
          f"l2_lambda={params['l2_lambda']}, learning_rate={params['learning_rate']}, "
          f"batch_size={params['batch_size']}")

    return params
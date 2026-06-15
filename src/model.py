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
#   Hidden layer 1 → 64 neurons, ReLU, L2 regularization
#   Dropout 1      → 30% dropout
#   Hidden layer 2 → 32 neurons, ReLU, L2 regularization
#   Dropout 2      → 30% dropout
#   Output layer   → 1 neuron, Sigmoid (binary classification)
#
# Loss function : Binary Crossentropy
# Optimizer     : Adam
# Metric        : Accuracy (baseline) / Accuracy, AUC (optimized)
# ==============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ── Constants ──────────────────────────────────────────────────────────────────
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
        input_dim (int): Number of input features (e.g. 11 for WineQT.csv
                          after preprocessing)

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
                64,
                activation="relu",
                kernel_regularizer=regularizers.L2(l2_lambda),
                name="hidden_1",
            ),
            layers.Dropout(dropout_rate, name="dropout_1"),

            # ── Hidden layer 2 ─────────────────────────────────────────────────
            layers.Dense(
                32,
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
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model
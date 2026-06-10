# ==============================================================================
# src/model.py
# ------------------------------------------------------------------------------
# Defines the ANN architecture for the Wine Quality binary classification task.
#
# Functions:
#   build_model(input_dim, ...)  → compiles and returns a Keras Sequential model
#
# Architecture (baseline):
#   Input layer  → input_dim features (11)
#   Hidden layer 1 → 64 neurons, ReLU, L2 regularization
#   Dropout 1      → 30% dropout
#   Hidden layer 2 → 32 neurons, ReLU, L2 regularization
#   Dropout 2      → 30% dropout
#   Output layer   → 1 neuron, Sigmoid (binary classification)
#
# Loss function : Binary Crossentropy
# Optimizer     : Adam
# Metric        : Accuracy, AUC
# ==============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ── Constants ──────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.001   # Adam optimizer default learning rate
L2_LAMBDA     = 0.001   # L2 regularization strength
DROPOUT_RATE  = 0.3     # Fraction of neurons dropped during training


def build_model(
    input_dim: int,
    learning_rate: float = LEARNING_RATE,
    l2_lambda: float     = L2_LAMBDA,
    dropout_rate: float  = DROPOUT_RATE,
) -> keras.Sequential:
    """
    Build and compile the ANN model for binary wine quality classification.

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
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model
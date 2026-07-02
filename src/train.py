# ==============================================================================
# src/train.py
# ------------------------------------------------------------------------------
# Handles model training for the Wine Quality ANN pipeline.
#
# Functions:
#   train_model(model, X_train, y_train, X_val, y_val, ...)
#       → trains the model and returns the training history
#
# Callbacks:
#   EarlyStopping  → stops training when val_loss stops improving
#                    (prevents overfitting and saves time)
#   ModelCheckpoint → saves the best model weights during training
#                    (based on lowest val_loss)
#
# Training config:
#   Epochs    : 100 (EarlyStopping will likely trigger earlier)
#   Batch size: generic fallback default only (see BATCH_SIZE below) —
#               NOT the tuned value. main.py always passes the actual
#               batch_size from models/best_params.json (Step 4 grid
#               search); this constant is only used when train_model()
#               is called directly without an explicit batch_size,
#               e.g. from a notebook.
# ==============================================================================

import os
import numpy as np
from tensorflow import keras


# ── Constants ──────────────────────────────────────────────────────────────────
EPOCHS          = 100   # Maximum number of training epochs
BATCH_SIZE      = 32    # Generic fallback default, NOT the tuned value —
                        # see src/model.py:load_best_params()
PATIENCE        = 10    # EarlyStopping: epochs to wait before stopping
MODEL_SAVE_PATH = "models/best_model.keras"


def train_model(
    model: keras.Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int     = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int   = PATIENCE,
    save_path: str  = MODEL_SAVE_PATH,
) -> keras.callbacks.History:
    """
    Train the ANN model with EarlyStopping and ModelCheckpoint.

    Training stops automatically when validation loss stops improving.
    The best model weights (lowest val_loss) are saved to disk.

    Args:
        model      (keras.Sequential) : Compiled Keras model
        X_train    (np.ndarray)       : Scaled training features
        y_train    (np.ndarray)       : Training labels
        X_val      (np.ndarray)       : Scaled validation features
        y_val      (np.ndarray)       : Validation labels
        epochs     (int)              : Maximum number of epochs
        batch_size (int)              : Samples per gradient update
        patience   (int)              : EarlyStopping patience
        save_path  (str)              : Path to save the best model

    Returns:
        keras.callbacks.History: Training history object containing
                                 loss and metric values per epoch
    """
    # ── Callbacks ──────────────────────────────────────────────────────────────
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",    # Watch validation loss
        patience=patience,     # Stop after N epochs without improvement
        restore_best_weights=True,  # Roll back to best weights when stopping
        verbose=1,
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor="val_loss",    # Save when validation loss improves
        save_best_only=True,   # Only overwrite if this epoch is better
        verbose=1,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\n   Epochs (max) : {epochs}")
    print(f"   Batch size   : {batch_size}")
    print(f"   Patience     : {patience}")
    print(f"   Saving to    : {save_path}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1,
    )

    stopped_epoch = early_stopping.stopped_epoch
    if stopped_epoch > 0:
        print(f"\n   ⏹️  Early stopping triggered at epoch {stopped_epoch + 1}")
    else:
        print(f"\n   ✅ Training completed all {epochs} epochs")

    print(f"   💾 Best model saved → {save_path}")

    return history
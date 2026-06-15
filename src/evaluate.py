# ==============================================================================
# src/evaluate.py
# ------------------------------------------------------------------------------
# Handles model evaluation and visualizations for the Wine Quality ANN pipeline.
#
# Functions:
#   evaluate_model(model, X_test, y_test, history)
#       → prints classification report and saves all plots
#
#   plot_confusion_matrix(y_test, y_pred, title, save_path)
#       → confusion matrix heatmap
#
#   plot_roc_curve(y_test, y_prob, title, save_path)
#       → ROC curve with AUC score
#
#   plot_training_history(history, save_path)
#       → loss and accuracy curves over epochs
#
# Output:
#   All figures saved to reports/figures/
# ==============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from tensorflow import keras


# ── Constants ──────────────────────────────────────────────────────────────────
# Resolve to <project_root>/reports/figures regardless of the caller's
# current working directory (e.g. main.py at project root vs. a notebook
# running from notebooks/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "reports", "figures")
THRESHOLD   = 0.5   # Probability threshold for binary classification

os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi"    : 120,
    "figure.figsize": (10, 5),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def evaluate_model(
    model: keras.Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    history: keras.callbacks.History,
) -> None:
    """
    Run full evaluation of the trained model on the test set.

    Steps:
        1. Generate predictions
        2. Print classification report (precision, recall, F1, accuracy)
        3. Plot confusion matrix
        4. Plot ROC curve
        5. Plot training history (loss & accuracy curves)

    Args:
        model   (keras.Sequential)       : Trained Keras model
        X_test  (np.ndarray)             : Scaled test features
        y_test  (np.ndarray)             : True test labels
        history (keras.callbacks.History): Training history from model.fit()
    """
    # ── Predictions ────────────────────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # ── Classification report ──────────────────────────────────────────────────
    print("\n── Classification Report ─────────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Poor (0)", "Good (1)"]))

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        y_test, y_pred,
        title="Wine Quality ANN — Confusion Matrix",
        save_path=f"{FIGURES_DIR}/confusion_matrix.png",
    )
    plot_roc_curve(
        y_test, y_prob,
        title="Wine Quality ANN — ROC Curve",
        save_path=f"{FIGURES_DIR}/roc_curve.png",
    )
    plot_training_history(
        history,
        save_path=f"{FIGURES_DIR}/training_history.png",
    )


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    title: str  = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    The confusion matrix shows four outcomes:
        True Negative  (TN): correctly predicted poor wine
        True Positive  (TP): correctly predicted good wine
        False Negative (FN): good wine predicted as poor
        False Positive (FP): poor wine predicted as good

    Args:
        y_test     (np.ndarray) : True labels
        y_pred     (np.ndarray) : Predicted labels
        title      (str)        : Plot title
        save_path  (str)        : File path to save the figure
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Poor (0)", "Good (1)"],
        yticklabels=["Poor (0)", "Good (1)"],
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   💾 Saved → {save_path}")

    plt.show()


def plot_roc_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    title: str     = "ROC Curve",
    save_path: str = None,
) -> float:
    """
    Plot and save the ROC curve with AUC score.

    The ROC curve shows the trade-off between:
        True Positive Rate  (Recall)    — y-axis
        False Positive Rate (1-Specificity) — x-axis

    AUC interpretation:
        0.5 → random guessing (no skill)
        1.0 → perfect model

    Args:
        y_test    (np.ndarray) : True labels
        y_prob    (np.ndarray) : Predicted probabilities for class 1
        title     (str)        : Plot title
        save_path (str)        : File path to save the figure

    Returns:
        float: AUC score
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5,
            linestyle="--", label="Random guess")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   💾 Saved → {save_path}")

    plt.show()

    return roc_auc


def plot_training_history(
    history: keras.callbacks.History,
    save_path: str = None,
) -> None:
    """
    Plot and save training and validation loss and accuracy curves.

    These curves show how the model learned over epochs:
        - Loss curves    : training vs validation loss per epoch
        - Accuracy curves: training vs validation accuracy per epoch

    A growing gap between train and val curves indicates overfitting.
    EarlyStopping prevents this by halting training at the right time.

    Args:
        history   (keras.callbacks.History) : Returned by model.fit()
        save_path (str)                     : File path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Loss ───────────────────────────────────────────────────────────────────
    axes[0].plot(history.history["loss"],     label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title("Loss per Epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()

    # ── Accuracy ───────────────────────────────────────────────────────────────
    axes[1].plot(history.history["accuracy"],     label="Train accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[1].set_title("Accuracy per Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   💾 Saved → {save_path}")

    plt.show()
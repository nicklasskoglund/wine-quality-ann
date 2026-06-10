# ==============================================================================
# src/data_preprocessing.py
# ------------------------------------------------------------------------------
# Handles all data loading, cleaning, and preprocessing for the Wine Quality
# ANN pipeline.
#
# Functions:
#   load_data(filepath)            → loads raw CSV, creates binary label
#   preprocess(X_train, X_val, X_test) → scales features with StandardScaler
#
# Binary label:
#   quality >= 6  →  1  (good)
#   quality <  6  →  0  (poor)
#
# Train / Val / Test split:
#   70% train | 15% validation | 15% test
#   Stratified on target to preserve class distribution
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Constants ──────────────────────────────────────────────────────────────────
QUALITY_THRESHOLD = 6    # Wines scored >= 6 are labelled as "good" (1)
RANDOM_STATE      = 42   # Seed for reproducibility
VAL_SIZE          = 0.15 # 15% of total data used for validation
TEST_SIZE         = 0.15 # 15% of total data used for testing


def load_data(filepath: str):
    """
    Load raw CSV, create binary target label and return train/val/test splits.

    The dataset (WineQT.csv) contains red and white wines combined with
    physicochemical features and a quality score (3–8).

    Steps:
        1. Load CSV file
        2. Drop duplicate rows
        3. Create binary label: quality >= 6 → 1 (good), else → 0 (poor)
        4. Drop original quality column and any ID columns
        5. Split into train / val / test (stratified)

    Args:
        filepath (str): Path to the raw CSV file (e.g. data/raw/WineQT.csv)

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
               as pandas DataFrames / Series
    """
    # ── Load ───────────────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"   Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Drop duplicates ────────────────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    print(f"   Duplicates removed : {n_dropped}")

    # ── Drop ID column if present ──────────────────────────────────────────────
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
        print(f"   Dropped column     : Id")

    # ── Create binary label ────────────────────────────────────────────────────
    df["label"] = (df["quality"] >= QUALITY_THRESHOLD).astype(int)
    df = df.drop(columns=["quality"])

    good = df["label"].sum()
    poor = len(df) - good
    print(f"   Class distribution : {good:,} good (1) | {poor:,} poor (0)")

    # ── Split features & target ────────────────────────────────────────────────
    X = df.drop(columns=["label"])
    y = df["label"]

    # ── Train / temp split (70/30) ─────────────────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ── Val / test split (50/50 of the 30%) ───────────────────────────────────
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"   Train : {X_train.shape[0]:,} samples")
    print(f"   Val   : {X_val.shape[0]:,} samples")
    print(f"   Test  : {X_test.shape[0]:,} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Scale features using StandardScaler.

    Fit the scaler on training data only, then transform all three splits.
    This prevents data leakage from validation and test sets into training.

    StandardScaler transforms each feature to have:
        mean = 0
        standard deviation = 1

    Args:
        X_train (pd.DataFrame): Training features
        X_val   (pd.DataFrame): Validation features
        X_test  (pd.DataFrame): Test features

    Returns:
        tuple: X_train_scaled, X_val_scaled, X_test_scaled as np.ndarrays
    """
    # ── Fit on train, transform all ────────────────────────────────────────────
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    print(f"   Scaler fitted on {X_train_scaled.shape[0]:,} training samples")
    print(f"   Features scaled : {X_train_scaled.shape[1]}")

    return X_train_scaled, X_val_scaled, X_test_scaled
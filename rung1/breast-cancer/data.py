"""
rung1/data.py — load the breast-cancer data, split it, standardise it.

Why a separate file: every other file (baseline, network) must see
EXACTLY the same split, or the comparison is unfair. So the split lives
in one place and everyone imports it.
"""

import numpy as np                                   # arrays and the maths on them
from sklearn.datasets import load_breast_cancer      # the dataset, ships with sklearn, no download
from sklearn.model_selection import train_test_split # shuffles rows and cuts them into two sets


def load_data(seed=0):
    """Return X_train, y_train, X_val, y_val, X_test, y_test (standardised).

    seed: fixes the shuffle so the split is identical every run.
    """
    # --- 1. load -------------------------------------------------------
    data = load_breast_cancer()                      # a bundle: numbers, labels, feature names
    X, y = data.data, data.target                    # X: 569 samples × 30 features; y: 0 malignant, 1 benign

    # --- 2. split 70 / 15 / 15 ----------------------------------------
    # First cut: peel the TEST set off and never look at it again until the end.
    # test_size=0.15  -> 15% of rows go to test
    # random_state    -> same shuffle every run (reproducible)
    # stratify=y      -> each piece keeps the same benign/malignant ratio
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    # Second cut: take VALIDATION out of the remaining 85%.
    # 0.15 / 0.85 ≈ 0.176 of what's left = 15% of the original.
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.176, random_state=seed, stratify=y_tmp
    )

    # --- 3. standardise: each feature -> mean 0, std 1 -----------------
    # Features are on wildly different scales (area ~1000, smoothness ~0.1);
    # a network trains badly on that. Rescale each column.
    # axis=0 = "down the rows" -> one number per feature, shape (30,)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    # Use the TRAIN mean/std for all three sets. Computing them on the
    # whole dataset would leak test information into training — the
    # first thing a sceptic checks.
    # Broadcasting: a (30,) vector subtracts from every row of a (398, 30) matrix.
    X_train = (X_train - mu) / sd
    X_val = (X_val - mu) / sd
    X_test = (X_test - mu) / sd

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":                           # runs only when I run this file directly
    # --- 4. check ------------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print("train", X_train.shape, "val", X_val.shape, "test", X_test.shape)
    print("benign fraction per split:", y_train.mean().round(3), y_val.mean().round(3), y_test.mean().round(3))
    print("train feature means:", X_train.mean(axis=0).round(3))   # expect ~0
    print("train feature stds: ", X_train.std(axis=0).round(3))    # expect ~1
    assert X_train.shape[1] == 30                    # a belief, checked: 30 features survived
    # Note: 0.63 benign means a model that always says "benign" scores 63%.
    # That is the floor every real number has to beat.
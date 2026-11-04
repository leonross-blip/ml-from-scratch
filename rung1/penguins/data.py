"""
rung1/penguins/data.py — load, clean, split and standardise the Palmer penguins.

NOTE: this is the cleaned-up version (comments, docstring, constants), tidied
with Claude on 4 Nov 2026. The logic is the version I wrote that afternoon;
my from-scratch rebuild (not committed) produced identical output.

Data: palmerpenguins==0.1.6 — Gorman et al., Palmer Station LTER, CC0.
"""

import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split

# The four measurements used as features, and the label column.
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
LABEL = "species"

# 70 / 15 / 15 split. Validation is cut from what is left after test,
# so its fraction of the remainder is 0.15 / 0.85, not 0.15.
TEST_FRAC = 0.15
VAL_FRAC = 0.15


def load_data(seed=0):
    """Return X_train, X_val, X_test, y_train, y_val, y_test, names.

    X arrays: (n, 4) floats, standardised with the TRAINING mean and SD only.
    y arrays: (n,) ints — 0 Adelie, 1 Gentoo, 2 Chinstrap (names[i] is code i).
    Split is stratified, so each part keeps the ~44 / 36 / 20 % species mix.
    """
    df = load_penguins()

    # Clean: select the columns we use FIRST, then drop missing rows.
    # (Dropping on the full table loses 9 extra birds for a missing `sex`:
    # 333 rows instead of 342.)
    data = df[[LABEL] + FEATURES].dropna()

    # Features and labels. X[i] and y[i] are the same penguin (same row order).
    X = data[FEATURES].to_numpy()
    y, names = pd.factorize(data[LABEL])

    # Split twice, stratified on the labels being cut each time.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=VAL_FRAC / (1 - TEST_FRAC),
        random_state=seed, stratify=y_tmp,
    )

    # Standardise with training statistics only (no leakage from val/test).
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    X_train = (X_train - mu) / sd
    X_val = (X_val - mu) / sd
    X_test = (X_test - mu) / sd

    return X_train, X_val, X_test, y_train, y_val, y_test, names


if __name__ == "__main__":
    # Checks — expected: 238 / 52 / 52; species [105 85 48] / [23 19 10] / [23 19 10];
    # train mean ~0 and SD 1 exactly; val/test close to 0 and 1 but not exact.
    X_train, X_val, X_test, y_train, y_val, y_test, names = load_data()
    print("names:", list(names))
    for split, X_s, y_s in [("train", X_train, y_train),
                            ("val", X_val, y_val),
                            ("test", X_test, y_test)]:
        print(f"{split:>5}  shape {X_s.shape}  species {np.bincount(y_s)}")
        print(f"       mean {X_s.mean(axis=0).round(2)}  sd {X_s.std(axis=0).round(2)}")
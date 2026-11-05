"""Logistic-regression baseline for the penguins species task.

Same split, same standardisation, same 4 features as the net, so the
comparison is fair: the only difference is the net's hidden layer.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from data import load_data


def fit_baseline(X_train, y_train, max_iter=1000):
    """Fit multinomial logistic regression on the training split only.

    One (4, 3) weight matrix + (3,) bias -> softmax: the net with no hidden layer.
    With 3 classes sklearn uses softmax by default (no multi_class argument needed).
    """
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def majority_accuracy(y_train, y_eval):
    """Accuracy of always guessing the most common training species (the no-model floor)."""
    most_common = np.bincount(y_train).argmax()
    return np.mean(y_eval == most_common)


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, names = load_data()
    model = fit_baseline(X_train, y_train)

    # Val is for looking while building; test is looked at once and then frozen.
    print(f"majority  val  {majority_accuracy(y_train, y_val):.3f}")
    print(f"logistic  val  {model.score(X_val, y_val):.3f}")
    print(f"logistic  test {model.score(X_test, y_test):.3f}")

    # Which val penguins does it get wrong? (true -> predicted)
    wrong = np.where(model.predict(X_val) != y_val)[0]
    for i in wrong:
        print(f"  val row {i}: {names[y_val[i]]} -> {names[model.predict(X_val[i:i+1])[0]]}")
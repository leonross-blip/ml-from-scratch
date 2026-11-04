"""
rung1/train.py — train the NumPy network with gradient descent and compare it with the baseline.
"""

import numpy as np                                   # arrays and the maths on them
import matplotlib.pyplot as plt                      # the loss plot
from data import load_data                           # the same seed-0 split the baseline used -> fair comparison
from net import init_params, forward, loss, backward # start the weights, run the network, score it, get the gradients


def accuracy(params, X, y):
    """Fraction of rows where the network's prediction (p > 0.5) matches the label."""
    p, _ = forward(params, X)              # p (n,1) probabilities; _ = the cache, not needed here
    return np.mean((p[:, 0] > 0.5) == y)   # predict -> compare -> mean of True/False = fraction right


def train(X_train, y_train, X_val, y_val, epochs=3000, step=0.1, seed=0):
    """Full-batch gradient descent. Returns the parameters from the epoch with the lowest
    validation loss, that epoch, and the train/val loss for every epoch."""
    params = init_params(seed=seed)                       # tiny random weights, zero biases
    history = []                                          # one (train loss, val loss) pair per epoch
    best_val, best_epoch, best_params = np.inf, 0, None   # "best so far": anything beats infinity

    for epoch in range(epochs):                           # one epoch = one pass over all training rows
        p, cache = forward(params, X_train)               # probabilities, plus what backward needs
        grads = backward(params, p, cache, y_train)       # dL/dθ, a dict with the same keys as params
        for k in params:                                  # "W1", "b1", "W2", "b2"
            params[k] -= step * grads[k]                  # θ ← θ − α·dL/dθ

        train_loss = loss(p, y_train)                     # how well it fits the rows it learns from
        val_loss = loss(forward(params, X_val)[0], y_val) # how well it does on rows it never learns from
        history.append((train_loss, val_loss))

        if val_loss < best_val:                           # early stopping: remember the best epoch
            best_val, best_epoch = val_loss, epoch
            best_params = {k: v.copy() for k, v in params.items()}   # a snapshot, not a pointer

    return best_params, best_epoch, np.array(history)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print(f"before training: train accuracy {accuracy(init_params(), X_train, y_train):.3f}")

    params, best_epoch, history = train(X_train, y_train, X_val, y_val)
    print(f"best epoch {best_epoch}, val loss {history[best_epoch, 1]:.4f}")
    print(f"train accuracy {accuracy(params, X_train, y_train):.3f}")
    print(f"val accuracy   {accuracy(params, X_val, y_val):.3f}")
    print(f"test accuracy  {accuracy(params, X_test, y_test):.3f}   (baseline 0.965)")

    plt.plot(history[:, 0], label="train")
    plt.plot(history[:, 1], label="val")
    plt.axvline(best_epoch, linestyle="--", color="grey", label="best epoch")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.yscale("log")                                     # log scale: the late, small changes stay visible
    plt.legend()
    plt.savefig("loss.png", dpi=120)
    print("saved loss.png")
    

"""
rung1/baseline.py — logistic regression, the number the network must match.

Notation for the whole rung: x = the 30 standardised inputs, h = hidden
units (network only — this model has none), z = pre-sigmoid number, p = sigmoid(z).

The model, for one tumour with standardised features x1..x30:
    z = w1*x1 + w2*x2 + ... + w30*x30 + b        (31 parameters, no hidden layer)
    p = sigmoid(z) = 1 / (1 + e^-z)              (a probability in (0,1))
    predict benign if p > 0.5
Each row is one Bernoulli trial with its own p. Training = choose w, b to
maximise the Bernoulli likelihood over the training rows, i.e. minimise
binary cross-entropy  L = -mean( y*log p + (1-y)*log(1-p) ).
The network I build next keeps this exact last step but feeds it 16 hidden
units h = ReLU(W1 x + b1) instead of the 30 raw x's: same loss, same data,
31 parameters instead of 513. That is why it is the fair baseline.
"""

from sklearn.linear_model import LogisticRegression   # the model above, loss and optimiser built in
from data import load_data                            # same folder -> found first; imports the recipe only


def fit_baseline(X_train, y_train, max_iter=1000):
    """Fit logistic regression on the training pile; return the fitted model."""
    model = LogisticRegression(max_iter=max_iter)   # empty container with settings; nothing learnt yet
    model.fit(X_train, y_train)                     # gradient descent on L; learned w, b stored INSIDE
    return model                                    # model.coef_ = w (1,30), model.intercept_ = b (1,)


if __name__ == "__main__":
    # Variables do not cross files: importing gave me the recipe, calling it
    # makes the six piles here. Same seed everywhere -> identical split -> fair.
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    model = fit_baseline(X_train, y_train)

    # .score = for each row compute p, predict (p > 0.5), compare to y, return fraction right.
    print(f"val accuracy  {model.score(X_val, y_val):.3f}")    # for looking
    print(f"test accuracy {model.score(X_test, y_test):.3f}")  # looked at once; goes in README
    print(f"w {model.coef_.shape}, b {model.intercept_.shape}")  # (1, 30) and (1,): the 31 parameters

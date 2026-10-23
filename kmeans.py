"""K-means from scratch on the Palmer penguins (MLB L02).

Steps: load + standardise -> pick starting centres -> repeat
(assign each point to its nearest centre, move each centre to the
mean of its points) until no point changes cluster.

Done: K-means, WSS, best-of-N random restarts, choosing K
(WSS curve + average silhouette).
"""

import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


# ---------------------------------------------------------------- data

def load_penguins():
    """Return Z (n, 4) standardised features and the matching species labels."""
    df = pd.read_csv(URL)
    clean = df.dropna(subset=FEATURES)          # 344 -> 342: drop birds with no measurements
    species = clean["species"].to_numpy()       # labels, only for checking at the end
    X = clean[FEATURES].to_numpy()              # (342, 4) raw numbers
    Z = (X - X.mean(axis=0)) / X.std(axis=0)    # each column: mean 0, sd 1
    return Z, species


# ---------------------------------------------------------------- K-means pieces

def init_centres(Z, k, rng):
    """Pick k different data points as the starting centres -> (k, 4)."""
    rows = rng.choice(len(Z), size=k, replace=False)   # k different row numbers
    return Z[rows]


def assign(Z, centres):
    """ASSIGN step: return labels (n,), the index of each point's nearest centre."""
    D = []
    for p in Z:                                    # one penguin at a time
        row = []
        for c in centres:                          # each centre in turn
            row.append(((p - c) ** 2).sum())       # squared distance p -> c
        D.append(row)
    D = np.array(D)                                # (n, k) distance table
    return D.argmin(axis=1)                        # column of the smallest in each row


def update(Z, labels, k):
    """UPDATE step: move each centre to the mean of its points -> (k, 4)."""
    centres = []
    for j in range(k):                             # clusters 0 .. k-1
        members = Z[labels == j]                   # only cluster j's penguins
        centres.append(members.mean(axis=0))       # average down each column
    return np.array(centres)


def wss(Z, labels, centres):
    """Within-cluster sum of squares: each point's squared distance to its own centre, summed."""
    return ((Z - centres[labels]) ** 2).sum()      # centres[labels]: each point's own centre


# ---------------------------------------------------------------- the algorithm

def kmeans(Z, k, rng, max_iterations=100):
    """Run K-means once. Returns (centres, labels, n_rounds)."""
    centres = init_centres(Z, k, rng)
    previous_labels = np.array([])                 # empty: can never match round 1
    for i in range(max_iterations):
        labels = assign(Z, centres)
        if np.array_equal(previous_labels, labels):
            break                                  # nobody moved: converged
        centres = update(Z, labels, k)
        previous_labels = labels
    return centres, labels, i + 1


def best_of_restarts(Z, k, n_starts=10, species=None, verbose=True):
    """Run K-means from n_starts seeds; keep the run with the lowest WSS (no labels used)."""
    best = None
    for seed in range(n_starts):
        rng = np.random.default_rng(seed)          # one generator per seed: each run reproducible
        centres, labels, n_rounds = kmeans(Z, k, rng)
        score = wss(Z, labels, centres)
        if verbose:
            line = f"seed {seed}: WSS {score:.1f}, {n_rounds} rounds"
            if species is not None:
                line += f", agreement {agreement(species, labels):.1%}"
            print(line)
        if best is None or score < best["wss"]:
            best = {"seed": seed, "wss": score, "labels": labels, "centres": centres}
    return best


# ---------------------------------------------------------------- checking

def agreement(species, labels):
    """Share of points whose cluster's majority species is their own species."""
    table = pd.crosstab(species, labels)
    return table.max(axis=0).sum() / len(labels)


def distance_table(Z):
    """D[i, j] = plain (Euclidean) distance from point i to point j -> (n, n)."""
    diff = Z[:, None, :] - Z[None, :, :]           # (n, 1, 4) - (1, n, 4) -> (n, n, 4)
    return np.sqrt((diff ** 2).sum(axis=2))        # sum over the 4 features, then square root


def silhouette(D, labels):
    """Average silhouette over all points (-1 bad .. 1 good). D from distance_table."""
    s = []
    for i in range(len(labels)):
        same = labels == labels[i]                 # True for i's whole cluster
        same[i] = False                            # ...except i itself
        if same.sum() == 0:                        # i is alone in its cluster
            s.append(0.0)                          # convention: s = 0
            continue
        a = D[i][same].mean()                      # mean distance to own cluster

        b = min(D[i][labels == c].mean()           # mean distance to each other cluster,
                for c in np.unique(labels)         # keep the nearest one
                if c != labels[i])

        s.append((b - a) / max(a, b))
    return np.mean(s)


# ---------------------------------------------------------------- run

if __name__ == "__main__":
    Z, species = load_penguins()
    print("Z:", Z.shape)                                       # (342, 4)

    # 1. Ten restarts at K = 3, keep the lowest WSS.
    print("\nRestarts at K = 3:")
    best = best_of_restarts(Z, 3, n_starts=10, species=species)
    print(f"Best: seed {best['seed']}, WSS {best['wss']:.1f}, "
          f"agreement {agreement(species, best['labels']):.1%}")
    print(pd.crosstab(species, best["labels"]))

    # 2. Choosing K: WSS always falls; the silhouette should peak.
    D = distance_table(Z)                                      # (342, 342), built once
    print("\nK   WSS     silhouette")
    for k in range(2, 9):
        best = best_of_restarts(Z, k, n_starts=10, verbose=False)
        sil = silhouette(D, best["labels"])
        print(f"{k}   {best['wss']:6.1f}  {sil:.3f}")
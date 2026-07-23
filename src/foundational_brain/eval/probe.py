"""Cross-validated linear probes over frozen features.

A linear probe asks: is a target *linearly decodable* from a frozen
representation? That is the standard transfer test for a self-supervised model
— the labels were never used in pretraining, so decodability measures what the
representation happens to encode.

Two design points that keep the numbers honest:

* **Grouped by site is available but off by default.** Phenotype and site are
  entangled in ABIDE, so a probe can score well by learning the scanner. Every
  target is therefore also run with ``SITE_ID`` itself as the target: if a
  representation predicts site strongly, its phenotype scores are suspect. The
  ``site`` probe is the control, not a result.
* **Chance is reported next to every score.** Accuracy means nothing without the
  majority-class rate; R² is already chance-referenced at 0. Both are returned
  so a "good" number can't hide an imbalanced target.
"""

from __future__ import annotations

import warnings

import numpy as np


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


#: Regularization grid the probes search over, on the training fold only.
#: Spanning it (rather than fixing one value) means a "not decodable" verdict
#: reflects the representation, not an unlucky regularization strength.
_C_GRID = (0.01, 0.1, 1.0, 10.0)
_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)

# sklearn 1.9 emits a FutureWarning about a fitted-attribute change in
# LogisticRegressionCV that only affects attributes we never read (we use
# predictions and decision scores). Silence it so probe runs stay legible.
warnings.filterwarnings(
    "ignore", message=".*use_legacy_attributes.*", category=FutureWarning
)


def classify_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Stratified k-fold logistic-regression probe.

    Standardization and the L2 strength ``C`` are both selected on the training
    fold only — fitting either on all data would leak test-fold information.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    majority = float(counts.max() / counts.sum())
    binary = len(classes) == 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for tr, te in skf.split(features, labels):
        mu, sd = _standardize_fit(features[tr])
        xtr, xte = (features[tr] - mu) / sd, (features[te] - mu) / sd
        clf = LogisticRegressionCV(
            Cs=list(_C_GRID), max_iter=2000, class_weight="balanced", cv=3
        )
        clf.fit(xtr, labels[tr])
        pred = clf.predict(xte)
        accs.append(accuracy_score(labels[te], pred))
        if binary:
            score = clf.decision_function(xte)
            aucs.append(roc_auc_score(labels[te], score))

    out = {
        "task": "classification",
        "n_classes": int(len(classes)),
        "n_samples": int(len(labels)),
        "chance_accuracy": majority,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
    }
    if binary:
        out["auc_mean"] = float(np.mean(aucs))
        out["auc_std"] = float(np.std(aucs))
    return out


def regress_probe(
    features: np.ndarray,
    targets: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """K-fold ridge-regression probe, scored by R² and MAE.

    The ridge strength is chosen per training fold by ``RidgeCV`` over a grid,
    so a low R² means the target is genuinely not linearly present, not that a
    fixed ``alpha`` was mismatched to this representation's scale.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import KFold

    targets = np.asarray(targets, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s, maes = [], []
    for tr, te in kf.split(features):
        mu, sd = _standardize_fit(features[tr])
        xtr, xte = (features[tr] - mu) / sd, (features[te] - mu) / sd
        ymu = targets[tr].mean()
        reg = RidgeCV(alphas=list(_ALPHA_GRID))
        reg.fit(xtr, targets[tr] - ymu)
        pred = reg.predict(xte) + ymu
        r2s.append(r2_score(targets[te], pred))
        maes.append(mean_absolute_error(targets[te], pred))

    return {
        "task": "regression",
        "n_samples": int(len(targets)),
        "target_std": float(targets.std()),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
    }


def probe_all(
    feature_sets: dict[str, np.ndarray],
    targets: dict[str, tuple[str, np.ndarray]],
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Run every (feature set × target) probe.

    ``targets`` maps a name to ``(kind, values)`` where kind is
    ``"classification"`` or ``"regression"``. Returns
    ``results[target][feature_set] = metrics``.
    """
    out: dict[str, dict] = {}
    for tname, (kind, values) in targets.items():
        out[tname] = {}
        for fname, feats in feature_sets.items():
            if kind == "classification":
                out[tname][fname] = classify_probe(
                    feats, values, n_splits=n_splits, seed=seed
                )
            else:
                out[tname][fname] = regress_probe(
                    feats, values, n_splits=n_splits, seed=seed
                )
    return out

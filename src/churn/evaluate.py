"""Metrics and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def top_decile_lift(y_true, prob, frac: float = 0.1) -> float:
    """Churn rate among the top ``frac`` highest-risk customers divided by the base rate."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    if y_true.mean() == 0:
        return float("nan")
    n = max(1, int(round(len(y_true) * frac)))
    top = np.argsort(-prob)[:n]
    return float(y_true[top].mean() / y_true.mean())


def evaluate_model(pipe, X_test, y_test, threshold: float = 0.5) -> dict:
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "top_decile_lift": top_decile_lift(y_test, prob),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def feature_importance(pipe, top_n: int = 15) -> pd.DataFrame:
    """Importance (trees) or absolute coefficient (linear) per transformed feature."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_.ravel())
    else:
        raise ValueError("Model exposes neither feature_importances_ nor coef_")
    out = pd.DataFrame({"Feature": names, "Importance": values})
    return out.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(fitted: dict, X_test, y_test, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, pipe in fitted.items():
        prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(loc="lower right")
    _save(fig, out_dir / "roc_curves.png")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, pipe in fitted.items():
        prob = pipe.predict_proba(X_test)[:, 1]
        p, r, _ = precision_recall_curve(y_test, prob)
        ax.plot(r, p, label=f"{name} (AP {average_precision_score(y_test, prob):.3f})")
    ax.axhline(float(np.mean(y_test)), color="k", ls="--", label="Base rate")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    ax.legend(loc="upper right")
    _save(fig, out_dir / "pr_curves.png")


def plot_confusion(name: str, cm: list[list[int]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix: {name}")
    _save(fig, out_dir / "confusion_matrix.png")


def plot_importance(name: str, imp: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=imp, x="Importance", y="Feature", color="#2F5597", ax=ax)
    ax.set_title(f"Top drivers of churn: {name}")
    _save(fig, out_dir / "feature_importance.png")

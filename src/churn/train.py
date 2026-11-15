"""Train, compare and evaluate churn models.

Usage:
    python -m churn.train --data data/raw/customer_churn_data.xlsx
    python -m churn.train --demo            # synthetic data, for trying the pipeline only
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import evaluate as ev
from .config import DEFAULT_DATA_PATH, PROJECT_ROOT, RANDOM_STATE
from .data import load_data
from .features import column_groups, prepare_xy
from .models import build_models
from .synthetic import make_synthetic

README_START, README_END = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"


def results_markdown(results: dict, meta: dict) -> str:
    lines = []
    if meta["demo"]:
        lines.append("> **Synthetic demo data. These numbers are not real results.**\n")
    lines.append(
        f"{meta['n_rows']:,} customers, {meta['churn_rate']:.1%} churn, "
        f"stratified {1 - meta['test_size']:.0%}/{meta['test_size']:.0%} train/test split, "
        f"seed {meta['seed']}. Decision threshold 0.5.\n"
    )
    lines.append("| Model | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | Precision | Recall | F1 | Top-decile lift |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['cv_roc_auc_mean']:.3f} ± {r['cv_roc_auc_std']:.3f} | "
            f"{r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {r['top_decile_lift']:.2f}x |"
        )
    lines.append(f"\nBest model by cross-validated ROC-AUC: **{meta['best_model']}**.")
    return "\n".join(lines)


def update_readme(readme: Path, block: str) -> bool:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        return False
    new = f"{README_START}\n{block}\n{README_END}"
    readme.write_text(pattern.sub(lambda _: new, text), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="xlsx / csv / parquet file")
    p.add_argument("--demo", action="store_true", help="use synthetic data (metrics are meaningless)")
    p.add_argument("--reports-dir", type=Path, default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--no-xgboost", action="store_true")
    p.add_argument("--update-readme", action="store_true", help="write the results table into README.md")
    args = p.parse_args(argv)

    reports = args.reports_dir or PROJECT_ROOT / "reports" / ("demo" if args.demo else "")
    models_dir = args.models_dir or PROJECT_ROOT / "models" / ("demo" if args.demo else "")
    figures = reports / "figures"

    raw = make_synthetic(2000, seed=args.seed) if args.demo else load_data(args.data)
    if args.demo:
        print("!! Using SYNTHETIC demo data. Results are for smoke-testing only.")
    print(f"Loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns")

    X, y = prepare_xy(raw)
    print(f"After cleaning: {len(X):,} rows kept (none dropped for missing features), "
          f"{X.shape[1]} features, churn rate {y.mean():.2%}")
    numeric, categorical = column_groups(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = build_models(numeric, categorical, pos_weight, not args.no_xgboost, args.seed)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    results, fitted = {}, {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        metrics = ev.evaluate_model(pipe, X_test, y_test)
        metrics.update(cv_roc_auc_mean=float(scores.mean()), cv_roc_auc_std=float(scores.std()))
        results[name], fitted[name] = metrics, pipe
        print(f"{name:20s} CV AUC {scores.mean():.3f} | test AUC {metrics['roc_auc']:.3f} "
              f"| PR-AUC {metrics['pr_auc']:.3f} | recall {metrics['recall']:.3f}")

    # Model selection uses cross-validation on the training split only, never the test set.
    best = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    meta = {
        "demo": args.demo, "n_rows": int(len(X)), "churn_rate": float(y.mean()),
        "test_size": args.test_size, "seed": args.seed, "best_model": best,
        "features": {"numeric": numeric, "categorical": categorical},
    }

    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    ev.plot_roc_pr(fitted, X_test, y_test, figures)
    ev.plot_confusion(best, results[best]["confusion_matrix"], figures)
    tree_name = next((n for n in ("XGBoost", "Random Forest") if n in fitted), best)
    imp = ev.feature_importance(fitted[tree_name])
    ev.plot_importance(tree_name, imp, figures)
    imp.to_csv(reports / "feature_importance.csv", index=False)

    (reports / "metrics.json").write_text(json.dumps({"meta": meta, "models": results}, indent=2))
    block = results_markdown(results, meta)
    (reports / "metrics.md").write_text(block + "\n", encoding="utf-8")
    joblib.dump({"model": fitted[best], "name": best, "meta": meta}, models_dir / "best_model.joblib")
    print(f"\nBest model (CV ROC-AUC): {best}")
    print(f"Wrote metrics, figures and model to {reports} and {models_dir}")

    if args.update_readme:
        if args.demo:
            print("Refusing to write synthetic results into the README.")
        elif update_readme(PROJECT_ROOT / "README.md", block):
            print("README.md results table updated.")
        else:
            print("README.md has no RESULTS markers; nothing updated.")
    return {"meta": meta, "models": results, "importance": imp}


if __name__ == "__main__":
    main()

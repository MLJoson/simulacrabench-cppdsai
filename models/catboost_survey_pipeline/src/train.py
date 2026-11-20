from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from common import (
    load_schema, class_columns, check_columns, prepare_features,
    make_splits, target_mask, gate_audit
)


def safe_metrics(y_true, y_pred):
    labels = sorted(set(y_true.astype(str)) | set(y_pred.astype(str)))
    return {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_true_classes": int(len(set(y_true))),
        "n_predicted_classes": int(len(set(y_pred))),
        "labels": labels
    }


def train_dataset(name: str, parquet_path: str, schema_path: str,
                  output_dir: str, seed: int, params: dict):
    out = Path(output_dir) / name
    models_dir = out / "models"
    reports_dir = out / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Loading {name} ===")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    schema = load_schema(schema_path)
    check_columns(df, schema, name)

    given = class_columns(schema, "GIVEN")
    targets = class_columns(schema, "PREDICT")
    excluded = class_columns(schema, "EXCLUDE")
    print(f"Rows: {len(df):,} | GIVEN: {len(given)} | PREDICT: {len(targets)} | EXCLUDE: {len(excluded)}")
    print(f"Using only these {len(given)} GIVEN columns as features.")

    X_all = prepare_features(df, given)
    split = make_splits(len(df), schema["split"], seed)
    split.to_csv(reports_dir / "row_splits.csv", index=False)

    metrics_rows = []
    gate_rows = []
    skipped_rows = []

    for i, target in enumerate(targets, start=1):
        print(f"[{name}] {i}/{len(targets)}: {target}")
        y_all = df[target]
        observed = target_mask(y_all, schema["gated_value"])
        eligible_train = (split == "train").to_numpy() & observed.to_numpy()
        eligible_dev = (split == "dev").to_numpy() & observed.to_numpy()
        eligible_test = (split == "test").to_numpy() & observed.to_numpy()

        n_train = int(eligible_train.sum())
        n_dev = int(eligible_dev.sum())
        n_test = int(eligible_test.sum())
        gate_rows.append(gate_audit(df, schema, target))

        if n_train < 2:
            skipped_rows.append({"target": target, "reason": "Fewer than 2 observed training rows", "n_train": n_train})
            continue

        y_train = y_all.loc[eligible_train].astype(str)
        y_dev = y_all.loc[eligible_dev].astype(str)
        y_test = y_all.loc[eligible_test].astype(str)
        train_classes = set(y_train.unique())

        if len(train_classes) < 2:
            skipped_rows.append({"target": target, "reason": "Only one class in training data", "n_train": n_train})
            continue

        # CatBoost can predict only classes seen in its training labels.
        # Dev/test rows with unseen labels are excluded from the scored subset and counted.
        dev_known = y_dev.isin(train_classes)
        test_known = y_test.isin(train_classes)
        if (~dev_known).any():
            warnings.warn(f"{name}.{target}: {int((~dev_known).sum())} dev rows have labels absent from train; excluded from dev evaluation.")
        if (~test_known).any():
            warnings.warn(f"{name}.{target}: {int((~test_known).sum())} test rows have labels absent from train; excluded from test evaluation.")

        model = CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=seed,
            allow_writing_files=False,
            **params
        )
        fit_args = {
            "X": X_all.loc[eligible_train],
            "y": y_train,
            "cat_features": list(range(len(given))),
            "verbose": False
        }
        if n_dev > 0 and dev_known.sum() > 0:
            fit_args["eval_set"] = (X_all.loc[eligible_dev][dev_known.to_numpy()], y_dev.loc[dev_known])
            fit_args["early_stopping_rounds"] = int(params.get("early_stopping_rounds", 100))
        model.fit(**fit_args)

        model_path = models_dir / f"{target}.cbm"
        model.save_model(str(model_path))

        row = {
            "dataset": name, "target": target,
            "n_train_observed": n_train, "n_dev_observed": n_dev, "n_test_observed": n_test,
            "n_train_classes": len(train_classes),
            "n_dev_unseen_classes": int((~dev_known).sum()),
            "n_test_unseen_classes": int((~test_known).sum()),
            "model_path": str(model_path)
        }

        if test_known.sum() > 0:
            pred = model.predict(X_all.loc[eligible_test][test_known.to_numpy()]).reshape(-1).astype(str)
            result = safe_metrics(y_test.loc[test_known].to_numpy(), pred)
            row.update({k: v for k, v in result.items() if k != "labels"})
            report = classification_report(
                y_test.loc[test_known].to_numpy(), pred,
                output_dict=True, zero_division=0
            )
            pd.DataFrame(report).T.to_csv(reports_dir / f"{target}_classification_report.csv")
            cm_labels = sorted(set(y_test.loc[test_known].astype(str)) | set(pred))
            cm = confusion_matrix(y_test.loc[test_known].astype(str), pred, labels=cm_labels)
            pd.DataFrame(cm, index=cm_labels, columns=cm_labels).to_csv(
                reports_dir / f"{target}_confusion_matrix.csv"
            )
        else:
            row["accuracy"] = row["macro_f1"] = row["weighted_f1"] = np.nan
            row["macro_precision"] = row["macro_recall"] = np.nan
            row["n_test"] = 0
        metrics_rows.append(row)

    pd.DataFrame(metrics_rows).to_csv(reports_dir / "target_metrics.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(reports_dir / "gate_audit.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(reports_dir / "skipped_targets.csv", index=False)

    summary = {
        "dataset": name, "rows": len(df), "given_features": len(given),
        "predict_targets": len(targets), "trained_models": len(metrics_rows),
        "skipped_targets": len(skipped_rows),
        "split_counts": split.value_counts().to_dict(),
        "note": "Rows were assigned to train/dev/test using the schema counts and configured seed."
    }
    with open(reports_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Finished {name}: trained {len(metrics_rows)} / {len(targets)} targets.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train one CatBoost classifier per PREDICT item.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dataset", choices=["all", "unhcr", "unicef", "world_bank"], default="all")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    cb_params = dict(config["catboost"])
    cb_params.pop("early_stopping_rounds", None)
    # Early stopping is passed to fit(), not the model constructor.
    fit_params = dict(config["catboost"])
    fit_params.pop("early_stopping_rounds", None)

    selected = config["datasets"] if args.dataset == "all" else {args.dataset: config["datasets"][args.dataset]}
    summaries = []
    for name, spec in selected.items():
        summaries.append(train_dataset(
            name, spec["parquet"], spec["schema"],
            config["output_dir"], config["random_seed"], fit_params
        ))
    print("\nRun summaries:")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

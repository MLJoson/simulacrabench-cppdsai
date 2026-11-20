from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MISSING = "__MISSING__"


def load_schema(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    required = {"items", "split", "gated_value"}
    missing = required - set(schema)
    if missing:
        raise ValueError(f"Schema {path} is missing keys: {sorted(missing)}")
    return schema


def class_columns(schema: dict[str, Any], class_name: str) -> list[str]:
    return [
        col for col, meta in schema["items"].items()
        if meta.get("class") == class_name
    ]


def check_columns(df: pd.DataFrame, schema: dict[str, Any], dataset_name: str) -> None:
    schema_cols = list(schema["items"])
    missing = [c for c in schema_cols if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:20])
        raise ValueError(
            f"{dataset_name}: Parquet is missing {len(missing)} schema columns. "
            f"First missing columns: {preview}"
        )
    extra = [c for c in df.columns if c not in schema["items"]]
    if extra:
        print(f"[{dataset_name}] Note: ignoring {len(extra)} columns not declared in schema.")


def prepare_features(df: pd.DataFrame, given_cols: list[str]) -> pd.DataFrame:
    """CatBoost categorical inputs. Only schema-classified GIVEN columns are used."""
    X = df.loc[:, given_cols].copy()
    for col in given_cols:
        # CatBoost categorical columns must not contain NaN and should be strings.
        X[col] = X[col].map(
            lambda v: MISSING if pd.isna(v) else str(v)
        ).astype("object")
    return X


def make_splits(n_rows: int, split_spec: dict, seed: int) -> pd.Series:
    """Create one deterministic row split shared by every target in this dataset."""
    n_train = int(split_spec["n_train"])
    n_dev = int(split_spec["n_dev"])
    n_test = int(split_spec["n_test"])
    if n_train + n_dev + n_test != n_rows:
        raise ValueError(
            f"Schema split totals {n_train+n_dev+n_test}, but Parquet has {n_rows} rows. "
            "Check that this is the matching dataset/version."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    split = np.empty(n_rows, dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train:n_train+n_dev]] = "dev"
    split[order[n_train+n_dev:]] = "test"
    return pd.Series(split, index=np.arange(n_rows), name="split")


def target_mask(y: pd.Series, gated_value: str) -> pd.Series:
    # Do not drop substantive answers such as "Don't know" or "Prefer not to answer".
    return y.notna() & y.astype("string").ne(str(gated_value))


def gate_audit(df: pd.DataFrame, schema: dict, target: str) -> dict:
    meta = schema["items"][target]
    gate = meta.get("gate")
    if not gate:
        return {"target": target, "has_gate": False}
    parent = gate.get("parent")
    observed_if = [str(v) for v in gate.get("observed_if", [])]
    if parent not in df.columns:
        return {
            "target": target, "has_gate": True, "parent": parent,
            "observed_if": "|".join(observed_if),
            "status": "PARENT_COLUMN_MISSING"
        }
    parent_values = df[parent].astype("string")
    expected = parent_values.isin(observed_if)
    y = df[target]
    actual = target_mask(y, schema["gated_value"])
    # This is an audit, not a forced overwrite: real survey routing may differ.
    return {
        "target": target,
        "has_gate": True,
        "parent": parent,
        "observed_if": "|".join(observed_if),
        "expected_eligible_rows": int(expected.sum()),
        "actually_observed_rows": int(actual.sum()),
        "expected_but_gated_rows": int((expected & ~actual).sum()),
        "observed_outside_expected_gate_rows": int((~expected & actual).sum()),
        "gate_match_rate": float((expected == actual).mean())
    }

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from common import load_schema, class_columns, MISSING


def normalize_feature_row(given_answers: dict, given_cols: list[str]) -> pd.DataFrame:
    row = {}
    for col in given_cols:
        value = given_answers.get(col, None)
        row[col] = MISSING if value is None or pd.isna(value) else str(value)
    return pd.DataFrame([row], columns=given_cols)


def predict_one_dataset(dataset: str, schema_path: str, models_dir: str,
                        given_answers: dict) -> dict:
    schema = load_schema(schema_path)
    items = schema["items"]
    given_cols = class_columns(schema, "GIVEN")
    predict_cols = class_columns(schema, "PREDICT")
    feature_row = normalize_feature_row(given_answers, given_cols)
    model_cache = {}
    result = {}
    resolving = set()

    def predict_target(target: str):
        if target in result:
            return result[target]
        if target in resolving:
            raise ValueError(f"Cycle found in gate dependencies at {target}")
        resolving.add(target)

        gate = items[target].get("gate")
        if gate:
            parent = gate.get("parent")
            allowed = [str(v) for v in gate.get("observed_if", [])]
            parent_class = items.get(parent, {}).get("class")
            if parent_class == "GIVEN":
                parent_value = given_answers.get(parent)
            elif parent_class == "PREDICT":
                parent_result = predict_target(parent)
                parent_value = parent_result.get("prediction")
            else:
                raise ValueError(
                    f"Cannot resolve gate for {target}: parent {parent!r} "
                    f"is classified {parent_class!r}, not GIVEN or PREDICT."
                )
            if parent_value is None or str(parent_value) not in allowed:
                result[target] = {"prediction": schema["gated_value"], "status": "NOT_APPLICABLE"}
                resolving.remove(target)
                return result[target]

        model_path = Path(models_dir) / "models" / f"{target}.cbm"
        if not model_path.exists():
            result[target] = {"prediction": None, "status": "MODEL_NOT_FOUND"}
            resolving.remove(target)
            return result[target]

        if target not in model_cache:
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            model_cache[target] = model
        model = model_cache[target]
        pred = str(model.predict(feature_row).reshape(-1)[0])
        probs = model.predict_proba(feature_row)[0]
        classes = [str(c) for c in model.classes_]
        confidence = float(max(probs)) if len(probs) else None
        result[target] = {
            "prediction": pred,
            "confidence": confidence,
            "status": "PREDICTED"
        }
        resolving.remove(target)
        return result[target]

    for target in predict_cols:
        predict_target(target)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict PREDICT targets from GIVEN answers.")
    parser.add_argument("--dataset", choices=["unhcr", "unicef", "world_bank"], required=True)
    parser.add_argument("--answers", required=True, help="JSON file containing GIVEN question answers.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="predictions.json")
    args = parser.parse_args()

    config = json.load(open(args.config, "r", encoding="utf-8"))
    spec = config["datasets"][args.dataset]
    with open(args.answers, "r", encoding="utf-8") as f:
        answers = json.load(f)

    model_root = Path(config["output_dir"]) / args.dataset
    predictions = predict_one_dataset(args.dataset, spec["schema"], model_root, answers)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(predictions)} target results to {args.output}")


if __name__ == "__main__":
    main()

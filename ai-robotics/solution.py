#!/usr/bin/env python3
"""Reference solution for AI Robotics – strategy recognition.

Usage:
    python solution.py --data-dir . --output-dir .

Expected files in data-dir:
    train.csv
    test.csv
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

RANDOM_STATE = 42
ALLOWED_STRATEGIES = {"explorer", "collector", "guardian", "sprinter"}
TARGET_COL = "strategy_label"
ID_COL = "robot_id"


@dataclass
class DatasetDiagnostics:
    train_shape: Tuple[int, int]
    test_shape: Tuple[int, int]
    train_columns: List[str]
    test_columns: List[str]
    train_dtypes: Dict[str, str]
    test_dtypes: Dict[str, str]
    train_missing: Dict[str, int]
    test_missing: Dict[str, int]
    class_distribution: Dict[str, int]



def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)



def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv in {data_dir.resolve()}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df



def inspect_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> DatasetDiagnostics:
    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in train.csv")
    if TARGET_COL in test_df.columns:
        raise ValueError(f"Column '{TARGET_COL}' must not be present in test.csv")
    if ID_COL not in train_df.columns or ID_COL not in test_df.columns:
        raise ValueError(f"Both train.csv and test.csv must include '{ID_COL}'")

    invalid_labels = set(train_df[TARGET_COL].dropna().unique()) - ALLOWED_STRATEGIES
    if invalid_labels:
        raise ValueError(f"Unexpected strategy labels in train.csv: {sorted(invalid_labels)}")

    diagnostics = DatasetDiagnostics(
        train_shape=train_df.shape,
        test_shape=test_df.shape,
        train_columns=train_df.columns.tolist(),
        test_columns=test_df.columns.tolist(),
        train_dtypes={k: str(v) for k, v in train_df.dtypes.to_dict().items()},
        test_dtypes={k: str(v) for k, v in test_df.dtypes.to_dict().items()},
        train_missing={k: int(v) for k, v in train_df.isna().sum().to_dict().items()},
        test_missing={k: int(v) for k, v in test_df.isna().sum().to_dict().items()},
        class_distribution={k: int(v) for k, v in train_df[TARGET_COL].value_counts().to_dict().items()},
    )
    return diagnostics



def compute_subtasks_1_to_4(train_df: pd.DataFrame) -> Dict[int, object]:
    required = ["arena_type", "avg_speed_mps", "items_collected"]
    missing_required = [c for c in required if c not in train_df.columns]
    if missing_required:
        raise ValueError(f"train.csv missing required columns for subtasks 1-4: {missing_required}")

    mode_values = train_df["arena_type"].mode(dropna=True)
    if mode_values.empty:
        raise ValueError("Cannot compute mode for arena_type (no valid values).")

    answers = {
        1: int(train_df["arena_type"].nunique(dropna=True)),
        2: float(train_df["avg_speed_mps"].max()),
        3: str(mode_values.iloc[0]),
        4: float(train_df["items_collected"].max()),
    }
    return answers



def build_pipelines(X: pd.DataFrame) -> Dict[str, Pipeline]:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    cat_ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ])

    preprocess_ohe = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_ohe_pipe, categorical_cols),
        ]
    )

    preprocess_ord = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_ord_pipe, categorical_cols),
        ]
    )

    pipelines = {
        "logreg": Pipeline([
            ("prep", preprocess_ohe),
            (
                "clf",
                LogisticRegression(
                    max_iter=1500,
                    random_state=RANDOM_STATE,
                    multi_class="multinomial",
                    n_jobs=None,
                ),
            ),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocess_ohe),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
        "hist_gb": Pipeline([
            ("prep", preprocess_ord),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_depth=None,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }
    return pipelines



def cross_validate_macro_f1(
    pipelines: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Dict[str, Dict[str, object]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results: Dict[str, Dict[str, object]] = {}

    for name, pipe in pipelines.items():
        fold_scores: List[float] = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = clone(pipe)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            score = f1_score(y_val, pred, average="macro")
            fold_scores.append(float(score))

        results[name] = {
            "fold_macro_f1": fold_scores,
            "mean_macro_f1": float(np.mean(fold_scores)),
            "std_macro_f1": float(np.std(fold_scores)),
        }
    return results



def create_submission(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    subtask_answers: Dict[int, object],
) -> pd.DataFrame:
    if len(predictions) != len(test_df):
        raise ValueError("Predictions length does not match number of test rows.")

    invalid_pred = set(pd.Series(predictions).unique()) - ALLOWED_STRATEGIES
    if invalid_pred:
        raise ValueError(f"Predictions contain invalid classes: {sorted(invalid_pred)}")

    rows = [
        {"robot_id": "GLOBAL", "subtaskID": 1, "answer": str(subtask_answers[1])},
        {"robot_id": "GLOBAL", "subtaskID": 2, "answer": str(subtask_answers[2])},
        {"robot_id": "GLOBAL", "subtaskID": 3, "answer": str(subtask_answers[3])},
        {"robot_id": "GLOBAL", "subtaskID": 4, "answer": str(subtask_answers[4])},
    ]

    task5 = pd.DataFrame(
        {
            "robot_id": test_df[ID_COL].astype(str),
            "subtaskID": 5,
            "answer": predictions.astype(str),
        }
    )

    submission = pd.concat([pd.DataFrame(rows), task5], ignore_index=True)
    return submission



def validate_submission(submission: pd.DataFrame, test_df: pd.DataFrame) -> None:
    expected_cols = ["robot_id", "subtaskID", "answer"]
    if submission.columns.tolist() != expected_cols:
        raise ValueError(f"submission.csv columns must be exactly {expected_cols}")

    global_rows = submission[(submission["robot_id"] == "GLOBAL") & (submission["subtaskID"].isin([1, 2, 3, 4]))]
    if len(global_rows) != 4:
        raise ValueError("submission.csv must contain exactly 4 GLOBAL rows for subtasks 1-4")
    if set(global_rows["subtaskID"].tolist()) != {1, 2, 3, 4}:
        raise ValueError("GLOBAL rows must contain subtaskID 1,2,3,4 each exactly once")

    task5 = submission[submission["subtaskID"] == 5]
    if len(task5) != len(test_df):
        raise ValueError("submission.csv must contain one subtaskID=5 row per test sample")

    counts = task5["robot_id"].value_counts()
    duplicates = counts[counts != 1]
    if not duplicates.empty:
        raise ValueError("Each test robot_id must appear exactly once for subtaskID=5")

    test_ids = set(test_df[ID_COL].astype(str))
    pred_ids = set(task5["robot_id"].astype(str))
    if test_ids != pred_ids:
        raise ValueError("robot_id values for subtaskID=5 do not exactly match test.csv robot_id")

    invalid_answers = set(task5["answer"].unique()) - ALLOWED_STRATEGIES
    if invalid_answers:
        raise ValueError(f"Invalid strategy labels in submission: {sorted(invalid_answers)}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Reference solution for AI Robotics task")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory with train.csv and test.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory to save outputs")
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds for macro-F1 validation")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(args.data_dir)
    diagnostics = inspect_data(train_df, test_df)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].astype(str)

    pipelines = build_pipelines(X_train)
    cv_results = cross_validate_macro_f1(pipelines, X_train, y_train, n_splits=args.n_splits)

    best_model_name = max(cv_results, key=lambda n: cv_results[n]["mean_macro_f1"])
    best_pipeline = clone(pipelines[best_model_name])
    best_pipeline.fit(X_train, y_train)
    predictions = best_pipeline.predict(test_df)

    subtask_answers = compute_subtasks_1_to_4(train_df)
    submission = create_submission(test_df, predictions, subtask_answers)
    validate_submission(submission, test_df)

    submission_path = args.output_dir / "submission.csv"
    metrics_path = args.output_dir / "metrics.json"

    submission.to_csv(submission_path, index=False)

    metrics = {
        "random_state": RANDOM_STATE,
        "best_model": best_model_name,
        "cv_results": cv_results,
        "subtasks_1_to_4": subtask_answers,
        "diagnostics": {
            "train_shape": diagnostics.train_shape,
            "test_shape": diagnostics.test_shape,
            "class_distribution": diagnostics.class_distribution,
            "train_missing_total": int(sum(diagnostics.train_missing.values())),
            "test_missing_total": int(sum(diagnostics.test_missing.values())),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Dataset diagnostics ===")
    print(f"train shape: {diagnostics.train_shape}")
    print(f"test shape:  {diagnostics.test_shape}")
    print(f"target distribution: {diagnostics.class_distribution}")
    print(f"total missing train/test: {sum(diagnostics.train_missing.values())}/{sum(diagnostics.test_missing.values())}")

    print("\n=== Subtasks 1-4 ===")
    print(f"1) unique arena_type: {subtask_answers[1]}")
    print(f"2) max avg_speed_mps: {subtask_answers[2]}")
    print(f"3) mode arena_type: {subtask_answers[3]}")
    print(f"4) max items_collected: {subtask_answers[4]}")

    print("\n=== Cross-validation (Macro-F1) ===")
    for name, info in cv_results.items():
        print(
            f"{name:>14}: mean={info['mean_macro_f1']:.5f} "
            f"std={info['std_macro_f1']:.5f} folds={np.round(info['fold_macro_f1'], 5).tolist()}"
        )
    print(f"Best model: {best_model_name}")

    print(f"\nSaved submission: {submission_path.resolve()}")
    print(f"Saved metrics:    {metrics_path.resolve()}")


if __name__ == "__main__":
    main()

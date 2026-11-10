"""Supervised-learning pipelines for churn risk and monthly-charge modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ClassificationResult:
    """Fitted churn classifier and held-out performance."""

    model: Pipeline
    accuracy: float
    roc_auc: float


@dataclass(frozen=True)
class RegressionResult:
    """Fitted charge model and held-out error metrics."""

    model: Pipeline
    mae: float
    rmse: float


def validate_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Require every modeling column before splitting or fitting."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(frame) < 20:
        raise ValueError("At least 20 rows are required for train/test evaluation.")


def build_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build a leakage-safe preprocessing graph from feature dtypes."""
    numeric = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in feature_columns if column not in numeric]
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def train_churn_classifier(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str = "Churn",
    test_size: float = 0.25,
    random_state: int = 42,
) -> ClassificationResult:
    """Train and evaluate a K-nearest-neighbors churn classifier."""
    validate_columns(frame, feature_columns + [target])
    y = frame[target].map({"Yes": 1, "No": 0}) if frame[target].dtype == object else frame[target].astype(int)
    if y.isna().any() or y.nunique() != 2:
        raise ValueError("The churn target must contain exactly two classes.")
    x_train, x_test, y_train, y_test = train_test_split(
        frame[feature_columns],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(frame, feature_columns, scale_numeric=True)),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return ClassificationResult(
        model=model,
        accuracy=float(accuracy_score(y_test, predictions)),
        roc_auc=float(roc_auc_score(y_test, probabilities)),
    )


def train_charge_regressor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str = "MonthlyCharge",
    test_size: float = 0.25,
    random_state: int = 42,
) -> RegressionResult:
    """Train and evaluate a random-forest monthly-charge model."""
    validate_columns(frame, feature_columns + [target])
    y = pd.to_numeric(frame[target], errors="coerce")
    if y.isna().any():
        raise ValueError("The charge target must be numeric and nonmissing.")
    x_train, x_test, y_train, y_test = train_test_split(
        frame[feature_columns],
        y,
        test_size=test_size,
        random_state=random_state,
    )
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(frame, feature_columns, scale_numeric=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return RegressionResult(
        model=model,
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
    )

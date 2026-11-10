"""Tests for churn-risk and charge-modeling pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import train_charge_regressor, train_churn_classifier


def sample_frame(rows: int = 120) -> pd.DataFrame:
    """Create deterministic telecom-like customer records."""
    rng = np.random.default_rng(42)
    tenure = rng.integers(1, 72, rows)
    monthly_charge = 35 + 0.8 * tenure + rng.normal(0, 4, rows)
    contract = np.where(tenure > 36, "Two year", "Month-to-month")
    churn = np.where((tenure < 24) & (monthly_charge > 45), "Yes", "No")
    return pd.DataFrame(
        {
            "Tenure": tenure,
            "MonthlyCharge": monthly_charge,
            "Contract": contract,
            "InternetService": rng.choice(["Fiber", "DSL"], rows),
            "Churn": churn,
        }
    )


def test_churn_classifier_returns_valid_metrics() -> None:
    """The classifier should train and report bounded held-out metrics."""
    result = train_churn_classifier(
        sample_frame(),
        ["Tenure", "MonthlyCharge", "Contract", "InternetService"],
    )
    assert 0 <= result.accuracy <= 1
    assert 0 <= result.roc_auc <= 1


def test_charge_regressor_returns_nonnegative_errors() -> None:
    """The regressor should train and report nonnegative held-out error."""
    result = train_charge_regressor(
        sample_frame(),
        ["Tenure", "Contract", "InternetService"],
    )
    assert result.mae >= 0
    assert result.rmse >= result.mae

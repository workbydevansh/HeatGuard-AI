from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .preprocessing import coerce_numeric_frame
from .utils import humanize_feature_name


def _unwrap_estimator(model):
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def get_feature_importance(
    artifact: dict,
    *,
    X_sample: Optional[pd.DataFrame] = None,
    y_sample: Optional[pd.Series] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    feature_names = list(artifact.get("feature_names", []))
    model = artifact.get("primary_model")
    estimator = _unwrap_estimator(model)

    if estimator is not None and hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        if len(values) == len(feature_names):
            importance = pd.DataFrame({"feature": feature_names, "importance": values})
            importance = importance.sort_values("importance", ascending=False)
            return importance.head(top_n).reset_index(drop=True)

    if model is not None and X_sample is not None and y_sample is not None and len(X_sample) > 20:
        X_ready = coerce_numeric_frame(X_sample, feature_names)
        result = permutation_importance(
            model,
            X_ready,
            y_sample,
            n_repeats=5,
            random_state=42,
            scoring="neg_root_mean_squared_error",
        )
        importance = pd.DataFrame(
            {"feature": feature_names, "importance": np.maximum(result.importances_mean, 0)}
        )
        importance = importance.sort_values("importance", ascending=False)
        return importance.head(top_n).reset_index(drop=True)

    return pd.DataFrame(columns=["feature", "importance"])


def explain_prediction(
    importance_df: pd.DataFrame,
    *,
    risk_level: str,
    predicted_wbt: float,
) -> str:
    if importance_df.empty:
        return (
            f"The forecast indicates {risk_level.lower()} risk near {predicted_wbt:.1f}°C. "
            "Feature attribution is limited because the selected model does not expose importances."
        )

    top_features = [humanize_feature_name(value) for value in importance_df["feature"].head(3)]
    joined = ", ".join(top_features[:-1]) + (f", and {top_features[-1]}" if len(top_features) > 1 else top_features[0])

    if risk_level in {"High", "Extreme"}:
        return (
            f"Risk is {risk_level.lower()} mainly because {joined} are elevated or trending upward "
            f"in the recent history used by the model."
        )
    if risk_level == "Moderate":
        return (
            f"Risk is moderate, with the model weighing {joined}. Conditions should be monitored "
            "because small humidity or temperature increases can move the forecast into high risk."
        )
    return (
        f"Risk is currently low. The model is most sensitive to {joined}, and these signals do not "
        "show a dangerous wet-bulb pattern for the selected horizon."
    )


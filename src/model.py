from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .config import MODEL_PATH, RANDOM_STATE


@dataclass
class ModelEvaluation:
    name: str
    model: Pipeline
    mae: float
    rmse: float
    r2: float


def build_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    candidates: dict[str, Any] = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=90,
            max_depth=18,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.055,
            l2_regularization=0.05,
            random_state=random_state,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=110,
            max_depth=20,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBRegressor

        candidates["XGBRegressor"] = XGBRegressor(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        candidates["LGBMRegressor"] = LGBMRegressor(
            n_estimators=320,
            learning_rate=0.045,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )
    except Exception:
        pass

    return {
        name: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
        for name, estimator in candidates.items()
    }


def evaluate_regression(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_select_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[ModelEvaluation, list[ModelEvaluation]]:
    evaluations: list[ModelEvaluation] = []
    for name, model in build_candidate_models(random_state).items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        preds = fitted.predict(X_val)
        metrics = evaluate_regression(y_val, preds)
        evaluations.append(
            ModelEvaluation(
                name=name,
                model=fitted,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
            )
        )

    best = min(evaluations, key=lambda item: item.rmse)
    return best, evaluations


def refit_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    fitted = clone(model)
    fitted.fit(X, y)
    return fitted


def save_model_artifact(artifact: dict, path=MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, path)


def load_model_artifact(path=MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = load(path)
    return patch_loaded_artifact(artifact)


def patch_loaded_artifact(artifact: dict) -> dict:
    """Patch small sklearn pickle differences across local/Colab versions."""
    def visit(obj: Any, seen: set[int]) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if hasattr(obj, "_fit_dtype") and not hasattr(obj, "_fill_dtype"):
            try:
                obj._fill_dtype = obj._fit_dtype
            except Exception:
                pass

        if isinstance(obj, dict):
            for value in obj.values():
                visit(value, seen)
        elif isinstance(obj, (list, tuple, set)):
            for value in obj:
                visit(value, seen)
        elif hasattr(obj, "steps"):
            for _, step in getattr(obj, "steps", []):
                visit(step, seen)
        elif hasattr(obj, "estimators_"):
            visit(getattr(obj, "estimators_"), seen)

    visit(artifact, set())
    return artifact

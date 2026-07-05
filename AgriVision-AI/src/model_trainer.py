"""Training, evaluation, and model selection for crop production prediction."""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains, tunes, and evaluates regression models."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.results: dict[str, dict] = {}
        self.best_model = None
        self.best_model_name = None

    def get_models(self) -> dict[str, object]:
        """Return dictionary of models to train."""
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
            "LightGBM": LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
            "CatBoost": CatBoostRegressor(iterations=200, random_seed=42, verbose=0),
        }

    def evaluate(self, y_true, y_pred) -> dict[str, float]:
        """Calculate RMSE, MAE, R² metrics."""
        # Clip predictions to non-negative
        y_pred = np.clip(y_pred, 0, None)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": rmse, "MAE": mae, "R2": r2}

    def train_all(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Train all models and return comparison DataFrame."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        models = self.get_models()
        rows = []

        for name, model in models.items():
            logger.info("Training %s", name)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = self.evaluate(y_test, preds)

            # Cross-validation R² (3-fold for speed)
            cv_r2 = cross_val_score(model, X_train, y_train, cv=3, scoring="r2").mean()

            rows.append({
                "Model": name,
                "RMSE": round(metrics["RMSE"], 2),
                "MAE": round(metrics["MAE"], 2),
                "R2": round(metrics["R2"], 4),
                "CV_R2": round(cv_r2, 4),
            })
            self.results[name] = {
                "model": model,
                "metrics": metrics,
                "cv_r2": cv_r2,
            }

        results_df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
        best_name = results_df.iloc[0]["Model"]
        self.best_model_name = best_name
        self.best_model = self.results[best_name]["model"]
        logger.info("Best model selected: %s", self.best_model_name)

        # Store split for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = list(X.columns)

        return results_df

    def tune_best_model(self) -> None:
        """Apply RandomizedSearchCV to the best model (Random Forest / XGBoost)."""
        if "Random Forest" in self.best_model_name or "XGBoost" in self.best_model_name:
            logger.info("Tuning %s", self.best_model_name)
            if "Random Forest" in self.best_model_name:
                param_dist = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }
                base = RandomForestRegressor(random_state=42, n_jobs=-1)
            else:
                param_dist = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 1.0],
                }
                base = XGBRegressor(random_state=42, verbosity=0)

            search = RandomizedSearchCV(
                base, param_dist, n_iter=10, cv=3, scoring="r2",
                random_state=42, n_jobs=-1, verbose=0
            )
            search.fit(self.X_train, self.y_train)
            self.best_model = search.best_estimator_
            logger.info("Best parameters: %s", search.best_params_)

    def save_best_model(self, feature_names: list[str]) -> str:
        """Save best model and feature names using joblib."""
        path = os.path.join(self.models_dir, "best_model.joblib")
        joblib.dump({
            "model": self.best_model,
            "feature_names": feature_names,
            "model_name": self.best_model_name,
        }, path)
        logger.info("Model saved to %s", path)
        return path

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Run inference with the best model."""
        return np.clip(self.best_model.predict(X), 0, None)

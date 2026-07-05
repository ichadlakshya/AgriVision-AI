"""SHAP-based explainability utilities for AgriVision AI."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shap


class ExplainabilityEngine:
    """Computes SHAP values and generates explanation artifacts."""

    def __init__(self, model, X_train: pd.DataFrame, feature_names: list[str], output_dir: str = "reports"):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.explainer = None
        self.shap_values = None

    def build_explainer(self, sample_size: int = 500):
        """Build SHAP TreeExplainer (fast for tree-based models)."""
        # Sample background for speed
        background = shap.sample(self.X_train, min(sample_size, len(self.X_train)))
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.KernelExplainer(self.model.predict, background)
        return self.explainer

    def compute_shap_values(self, X: pd.DataFrame, sample_size: int = 1000):
        """Compute SHAP values for a dataset sample."""
        if self.explainer is None:
            self.build_explainer()
        X_sample = X.sample(min(sample_size, len(X)), random_state=42)
        self.shap_values = self.explainer.shap_values(X_sample)
        self.X_sample = X_sample
        return self.shap_values, X_sample

    def get_feature_importance_df(self) -> pd.DataFrame:
        """Return mean absolute SHAP values as feature importance DataFrame."""
        if self.shap_values is None:
            raise ValueError("Run compute_shap_values first.")
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        fi_df = pd.DataFrame({
            "Feature": self.feature_names,
            "SHAP_Importance": mean_abs,
        }).sort_values("SHAP_Importance", ascending=False).reset_index(drop=True)
        return fi_df

    def explain_single_prediction(self, row: pd.DataFrame) -> dict[str, object]:
        """
        Explain a single prediction row.
        Returns predicted value, base value, and top contributing features.
        """
        if self.explainer is None:
            self.build_explainer()
        shap_vals = self.explainer.shap_values(row)
        if hasattr(self.explainer, "expected_value"):
            base = float(np.atleast_1d(self.explainer.expected_value)[0])
        else:
            base = 0.0

        prediction = float(self.model.predict(row)[0])
        contributions = dict(zip(self.feature_names, shap_vals[0]))
        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "prediction": max(0, prediction),
            "base_value": base,
            "top_features": sorted_contribs[:8],
        }

    def save_summary_plot(self) -> str:
        """Save SHAP summary (beeswarm) plot."""
        if self.shap_values is None:
            raise ValueError("Run compute_shap_values first.")
        path = os.path.join(self.output_dir, "shap_summary.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_sample, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path

    def save_importance_bar_plot(self) -> str:
        """Save SHAP feature importance bar plot."""
        fi_df = self.get_feature_importance_df().head(12)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(fi_df["Feature"][::-1], fi_df["SHAP_Importance"][::-1], color="#2196F3")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (SHAP)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "shap_importance.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path

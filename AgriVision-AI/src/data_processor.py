"""Data loading, cleaning, and feature engineering for AgriVision AI."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, cleaning, and feature engineering."""

    required_columns = {
        "State_Name",
        "District_Name",
        "Season",
        "Crop",
        "Crop_Year",
        "Area",
        "Production",
        "cat_crop",
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df_raw = None
        self.df = None
        self.encoders = {}

    def load_data(self) -> pd.DataFrame:
        """Load raw dataset from CSV."""
        if not pd.io.common.file_exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found: {self.filepath}")

        self.df_raw = pd.read_csv(self.filepath)
        missing_columns = sorted(self.required_columns.difference(self.df_raw.columns))
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

        self.df_raw["Season"] = self.df_raw["Season"].astype(str).str.strip()
        return self.df_raw.copy()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, handle edge cases."""
        df = df.drop_duplicates()
        # Remove rows where Area or Production is zero or negative
        df = df[(df["Area"] > 0) & (df["Production"] > 0)]
        df = df.reset_index(drop=True)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features from raw data.

        Features added:
        - Yield: Production per unit area
        - Decade: Decade bin for the crop year
        - State/District/Crop/Season productivity scores
        - Historical production averages
        """
        df = df.copy()

        # 1. Yield = Production / Area
        df["Yield"] = df["Production"] / df["Area"]

        # 2. Decade feature
        df["Decade"] = (df["Crop_Year"] // 10) * 10

        # 3. State productivity score: mean production per state
        state_prod = df.groupby("State_Name")["Production"].mean()
        df["State_Productivity_Score"] = df["State_Name"].map(state_prod)

        # 4. District productivity score
        district_prod = df.groupby("District_Name")["Production"].mean()
        df["District_Productivity_Score"] = df["District_Name"].map(district_prod)

        # 5. Crop popularity score: frequency of each crop across records
        crop_counts = df["Crop"].value_counts()
        df["Crop_Popularity_Score"] = df["Crop"].map(crop_counts)

        # 6. Seasonal productivity score
        season_prod = df.groupby("Season")["Production"].mean()
        df["Seasonal_Productivity_Score"] = df["Season"].map(season_prod)

        # 7. Historical production average: mean production per (State, Crop)
        hist_avg = df.groupby(["State_Name", "Crop"])["Production"].mean()
        df["Historical_Avg_Production"] = df.set_index(["State_Name", "Crop"]).index.map(hist_avg)

        # 8. Log-transform target for better model performance (keep original too)
        df["Log_Production"] = np.log1p(df["Production"])
        df["Log_Area"] = np.log1p(df["Area"])

        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode categorical columns for ML use."""
        df = df.copy()
        cat_cols = ["State_Name", "District_Name", "Season", "Crop", "cat_crop"]
        for col in cat_cols:
            if col not in df.columns:
                raise ValueError(f"Missing categorical column: {col}")
            le = LabelEncoder()
            df[f"{col}_Enc"] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        return df

    def get_features_and_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Return feature matrix X and target vector y for ML."""
        feature_cols = [
            "State_Name_Enc", "District_Name_Enc", "Crop_Year",
            "Season_Enc", "Crop_Enc", "cat_crop_Enc",
            "Area", "Log_Area",
            "Yield", "Decade",
            "State_Productivity_Score", "District_Productivity_Score",
            "Crop_Popularity_Score", "Seasonal_Productivity_Score",
            "Historical_Avg_Production",
        ]
        X = df[feature_cols].fillna(0)
        y = df["Production"]
        return X, y, feature_cols

    def full_pipeline(self) -> pd.DataFrame:
        """Run the complete data processing pipeline."""
        df = self.load_data()
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.encode_categoricals(df)
        self.df = df
        logger.info("Processed dataset with %s rows and %s columns", len(df), len(df.columns))
        return df

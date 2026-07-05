"""Standalone CLI for crop production prediction."""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_processor import DataProcessor
from runtime import configure_logging

# ─────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "crop_modified.csv")

BANNER = """
╔══════════════════════════════════════════════════════╗
║        🌾  AgriVision AI — Prediction Engine        ║
║     Crop Production Forecasting  |  v1.0             ║
╚══════════════════════════════════════════════════════╝
"""

logger = configure_logging(__name__)

# ─────────────────────────────────────────────────────────────────
def load_bundle():
    """Load trained model + encoders."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run python train.py first.")
    bundle = joblib.load(MODEL_PATH)
    logger.info("Model loaded: %s", bundle.get('model_name', 'Best Model'))
    return bundle


def load_reference_data():
    """Load dataset for feature computation and validation."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found at data/crop_modified.csv")
    proc = DataProcessor(DATA_PATH)
    df   = proc.full_pipeline()
    return df, proc


# ─────────────────────────────────────────────────────────────────
class Predictor:
    """Handles single and batch crop production predictions."""

    def __init__(self):
        self.bundle    = load_bundle()
        self.model     = self.bundle["model"]
        self.features  = self.bundle["feature_names"]
        self.df, self.proc = load_reference_data()

    # ── Build feature row from user inputs ───────────────────────
    def _build_input(self, state: str, district: str, crop: str,
                     season: str, year: int, area: float) -> pd.DataFrame:
        """
        Construct the feature vector for one prediction.
        Uses historical averages from the dataset for aggregate features.
        """
        df = self.df
        enc = self.proc.encoders

        if area <= 0:
            raise ValueError("Area must be greater than zero.")

        def safe_encode(col: str, val: str) -> int:
            le = enc.get(col)
            if le is None:
                return 0
            try:
                return int(le.transform([val])[0])
            except ValueError:
                # Unseen label → use most common class
                return int(le.transform([le.classes_[0]])[0])

        # Aggregate productivity features from historical data
        state_prod  = df[df["State_Name"] == state]["Production"].mean() if state in df["State_Name"].values else df["Production"].mean()
        dist_prod   = df[df["District_Name"] == district]["Production"].mean() if district in df["District_Name"].values else df["Production"].mean()
        crop_pop    = (df["Crop"] == crop).sum()
        season_prod = df[df["Season"] == season]["Production"].mean() if season in df["Season"].values else df["Production"].mean()
        hist_avg    = df[(df["State_Name"] == state) & (df["Crop"] == crop)]["Production"].mean()
        if np.isnan(hist_avg):
            hist_avg = df[df["Crop"] == crop]["Production"].mean()
        if np.isnan(hist_avg):
            hist_avg = df["Production"].mean()

        row = {
            "State_Name_Enc":              safe_encode("State_Name", state),
            "District_Name_Enc":           safe_encode("District_Name", district),
            "Crop_Year":                   year,
            "Season_Enc":                  safe_encode("Season", season),
            "Crop_Enc":                    safe_encode("Crop", crop),
            "cat_crop_Enc":                0,   # will be refined below
            "Area":                        area,
            "Log_Area":                    np.log1p(area),
            "Yield":                       hist_avg / area if area > 0 else 0,
            "Decade":                      (year // 10) * 10,
            "State_Productivity_Score":    state_prod,
            "District_Productivity_Score": dist_prod,
            "Crop_Popularity_Score":       crop_pop,
            "Seasonal_Productivity_Score": season_prod,
            "Historical_Avg_Production":   hist_avg,
        }

        # Infer cat_crop from data
        match = df[df["Crop"] == crop]["cat_crop"]
        if len(match):
            cat = match.mode().iloc[0]
            row["cat_crop_Enc"] = safe_encode("cat_crop", cat)

        missing_features = [feature for feature in self.features if feature not in row]
        if missing_features:
            raise ValueError(f"Model feature mismatch: missing {missing_features}")

        return pd.DataFrame([row])[self.features].fillna(0)

    # ── Single prediction ────────────────────────────────────────
    def predict_one(self, state: str, district: str, crop: str,
                    season: str, year: int, area: float) -> dict:
        """Return prediction dict with production, yield, and confidence band."""
        X    = self._build_input(state, district, crop, season, year, area)
        pred = float(max(0, self.model.predict(X)[0]))
        yld  = round(pred / area, 4) if area > 0 else 0

        # Simple confidence band: ±15% (proxy without prediction intervals)
        low  = round(pred * 0.85)
        high = round(pred * 1.15)

        # Historical average for context
        hist = self.df[(self.df["State_Name"] == state) & (self.df["Crop"] == crop)]["Production"]
        hist_avg = int(hist.mean()) if len(hist) else None
        diff_pct = round((pred - hist_avg) / hist_avg * 100, 1) if hist_avg else None

        return {
            "State":              state,
            "District":           district,
            "Crop":               crop,
            "Season":             season,
            "Year":               year,
            "Area":               area,
            "Predicted_Production": round(pred),
            "Yield_t_per_ha":     yld,
            "Confidence_Low":     int(low),
            "Confidence_High":    int(high),
            "Historical_Avg":     hist_avg,
            "vs_Historical_pct":  diff_pct,
        }

    def _predict_rows(self, inp: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Predict rows from a pre-loaded dataframe."""
        required = ["State_Name", "District_Name", "Crop", "Season", "Crop_Year", "Area"]
        missing = [c for c in required if c not in inp.columns]
        if missing:
            raise ValueError(f"Missing columns in input data: {missing}")

        results = []
        for i, row in inp.iterrows():
            r = self.predict_one(
                state    = str(row["State_Name"]),
                district = str(row["District_Name"]),
                crop     = str(row["Crop"]),
                season   = str(row["Season"]),
                year     = int(row["Crop_Year"]),
                area     = float(row["Area"]),
            )
            results.append(r)
            if verbose:
                print(f"  [{i+1}/{len(inp)}] {r['Crop']} in {r['State']} → {r['Predicted_Production']:,} tons")

        return pd.DataFrame(results)

    def predict_dataframe(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Predict for an in-memory dataframe."""
        return self._predict_rows(inp, verbose=False)

    # ── Batch prediction from CSV ────────────────────────────────
    def predict_batch(self, input_csv: str, output_csv: str) -> pd.DataFrame:
        """
        Predict for every row in input CSV.
        Required columns: State_Name, District_Name, Crop, Season, Crop_Year, Area
        """
        inp = pd.read_csv(input_csv)
        out_df = self._predict_rows(inp, verbose=True)
        out_df.to_csv(output_csv, index=False)
        logger.info("Predictions saved to %s", output_csv)
        return out_df

    # ── Interactive CLI mode ─────────────────────────────────────
    def interactive(self):
        """Step-by-step interactive prediction wizard."""
        print(BANNER)
        df = self.df

        # Show available options
        states   = sorted(df["State_Name"].unique().tolist())
        seasons  = sorted(df["Season"].unique().tolist())

        print("Available states:")
        for i, s in enumerate(states, 1):
            print(f"  {i:>2}. {s}")

        state_idx = int(input("\nEnter state number: ").strip()) - 1
        state     = states[state_idx]

        districts = sorted(df[df["State_Name"] == state]["District_Name"].unique().tolist())
        print(f"\nDistricts in {state}:")
        for i, d in enumerate(districts, 1):
            print(f"  {i:>2}. {d}")
        dist_idx  = int(input("Enter district number: ").strip()) - 1
        district  = districts[dist_idx]

        crops = sorted(df["Crop"].unique().tolist())
        print("\nAvailable crops (top 30 shown):")
        for i, c in enumerate(crops[:30], 1):
            print(f"  {i:>2}. {c}")
        crop_idx = int(input("Enter crop number: ").strip()) - 1
        crop     = crops[crop_idx]

        print("\nSeasons:", ", ".join(seasons))
        season = input("Enter season: ").strip()

        year = int(input("Enter crop year (e.g. 2024): ").strip())
        area = float(input("Enter area (hectares): ").strip())

        result = self.predict_one(state, district, crop, season, year, area)
        self._print_result(result)

    # ── Pretty-print result ──────────────────────────────────────
    def _print_result(self, r: dict):
        print("\n" + "═" * 54)
        print("  🌾  PREDICTION RESULT")
        print("═" * 54)
        print(f"  State    : {r['State']}")
        print(f"  District : {r['District']}")
        print(f"  Crop     : {r['Crop']}")
        print(f"  Season   : {r['Season']}")
        print(f"  Year     : {r['Year']}")
        print(f"  Area     : {r['Area']:,.0f} ha")
        print("─" * 54)
        print(f"  📦 Predicted Production : {r['Predicted_Production']:>12,} tons")
        print(f"  📊 Yield (t/ha)         : {r['Yield_t_per_ha']:>12.2f}")
        print(f"  📉 Confidence Band      : {r['Confidence_Low']:,} – {r['Confidence_High']:,} tons")
        if r["Historical_Avg"]:
            direction = "▲ above" if r["vs_Historical_pct"] > 0 else "▼ below"
            print(f"  📅 Historical Average   : {r['Historical_Avg']:>12,} tons")
            print(f"  📈 vs Historical        : {direction} by {abs(r['vs_Historical_pct'])}%")
        print("═" * 54)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AgriVision AI — Crop Production Predictor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--state",    type=str, help="State name")
    parser.add_argument("--district", type=str, help="District name")
    parser.add_argument("--crop",     type=str, help="Crop name")
    parser.add_argument("--season",   type=str, help="Season (Kharif/Rabi/Whole Year...)")
    parser.add_argument("--year",     type=int, help="Crop year (e.g. 2024)")
    parser.add_argument("--area",     type=float, help="Area in hectares")
    parser.add_argument("--input",    type=str, help="Input CSV for batch prediction")
    parser.add_argument("--output",   type=str, default="predictions_output.csv",
                        help="Output CSV path (batch mode)")
    args = parser.parse_args()
    try:
        predictor = Predictor()

        # Batch mode
        if args.input:
            print(BANNER)
            logger.info("Batch mode: %s", args.input)
            predictor.predict_batch(args.input, args.output)

        # CLI args mode
        elif all([args.state, args.district, args.crop, args.season, args.year, args.area]):
            print(BANNER)
            result = predictor.predict_one(
                args.state, args.district, args.crop,
                args.season, args.year, args.area
            )
            predictor._print_result(result)

        # Interactive wizard mode
        else:
            predictor.interactive()
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
    except Exception:
        logger.exception("Prediction CLI failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

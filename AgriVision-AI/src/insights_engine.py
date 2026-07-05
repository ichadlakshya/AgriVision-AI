"""Dynamic agricultural insight generation for AgriVision AI."""

from __future__ import annotations

import pandas as pd


class InsightsEngine:
    """Automatically generates data-driven agricultural insights."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def top_producing_states(self, n: int = 5) -> list[str]:
        """Generate insights about top producing states."""
        state_prod = self.df.groupby("State_Name")["Production"].sum()
        total = state_prod.sum()
        if total <= 0:
            return ["No production data available for state ranking."]

        top = state_prod.nlargest(n)
        insights = []
        for state, prod in top.items():
            pct = round(prod / total * 100, 1)
            insights.append(f"🌾 **{state}** contributed **{pct}%** of national production "
                            f"({int(prod):,} tons total).")
        return insights

    def production_trend(self) -> list[str]:
        """Detect multi-year production trends."""
        yearly = self.df.groupby("Crop_Year")["Production"].sum().sort_index()
        recent = yearly.tail(5)
        older = yearly.iloc[-10:-5]
        if len(older) == 0 or len(recent) == 0:
            return ["📈 Insufficient history to compute trend."]

        if older.mean() == 0:
            return ["📈 Insufficient baseline production history to compute trend."]

        recent_avg = recent.mean()
        older_avg = older.mean()
        pct_change = round((recent_avg - older_avg) / older_avg * 100, 1)
        direction = "increased" if pct_change > 0 else "decreased"
        symbol = "📈" if pct_change > 0 else "📉"
        insights = [
            f"{symbol} National crop production **{direction} by {abs(pct_change)}%** "
            f"comparing the last 5 years vs prior 5 years.",
            f"📅 Peak production year: **{int(yearly.idxmax())}** "
            f"({int(yearly.max()):,} tons).",
        ]
        return insights

    def high_yield_crops(self, n: int = 5) -> list[str]:
        """Identify highest-yield crops."""
        crop_yield = self.df.groupby("Crop")["Yield"].mean().dropna()
        top = crop_yield.nlargest(n)
        insights = []
        for crop, yld in top.items():
            insights.append(f"🌱 **{crop}** has the highest average yield: "
                            f"**{round(yld, 2)} tons/unit area**.")
        return insights

    def seasonal_performance(self) -> list[str]:
        """Analyse seasonal production patterns."""
        season_prod = self.df.groupby("Season")["Production"].sum().sort_values(ascending=False)
        total = season_prod.sum()
        if total <= 0:
            return ["No seasonal production data available."]

        insights = []
        for season, prod in season_prod.items():
            pct = round(prod / total * 100, 1)
            insights.append(f"🗓️ **{season}** season accounts for **{pct}%** of total production.")
        return insights

    def state_crop_leaders(self) -> list[str]:
        """Find leading crop for each top state."""
        top_states = self.df.groupby("State_Name")["Production"].sum().nlargest(5).index
        insights = []
        for state in top_states:
            state_df = self.df[self.df["State_Name"] == state]
            top_crop = state_df.groupby("Crop")["Production"].sum().idxmax()
            top_prod = state_df.groupby("Crop")["Production"].sum().max()
            insights.append(
                f"🏆 In **{state}**, the top crop is **{top_crop}** "
                f"({int(top_prod):,} tons)."
            )
        return insights

    def diversity_insights(self) -> list[str]:
        """Insights about crop diversity."""
        total_crops = self.df["Crop"].nunique()
        total_states = self.df["State_Name"].nunique()
        state_diversity = self.df.groupby("State_Name")["Crop"].nunique()
        if state_diversity.empty:
            return ["No diversity insights available."]

        most_diverse_state = state_diversity.idxmax()
        most_diverse_count = int(state_diversity.max())
        return [
            f"🌍 The dataset covers **{total_crops} unique crops** across **{total_states} states**.",
            f"🌿 **{most_diverse_state}** is the most diverse state, "
            f"cultivating **{most_diverse_count} different crops**.",
        ]

    def generate_all(self) -> dict[str, list[str]]:
        """Run all insight generators and return categorized results."""
        return {
            "Top Producing States": self.top_producing_states(),
            "Production Trends": self.production_trend(),
            "High-Yield Crops": self.high_yield_crops(),
            "Seasonal Performance": self.seasonal_performance(),
            "State-Crop Leaders": self.state_crop_leaders(),
            "Agricultural Diversity": self.diversity_insights(),
        }

"""Anthropic-backed agricultural chatbot with rule-based fallback."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from runtime import env, configure_logging

logger = configure_logging(__name__)


class AgriChatbot:
    """
    Professional agricultural chatbot powered by Claude (Anthropic API).
    Streams responses and maintains full conversation history.
    Falls back to rule-based answers if API key is unavailable.
    """

    ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, df: pd.DataFrame, api_key: str | None = None, model: str | None = None):
        self.df = df
        self.stats = self._compute_stats()
        self.system_prompt = self._build_system_prompt()
        self.api_key = api_key or env("ANTHROPIC_API_KEY", "") or ""
        self.model = model or env("ANTHROPIC_MODEL", self.DEFAULT_MODEL) or self.DEFAULT_MODEL

    # ── Statistics pre-computation ──────────────────────────────
    def _compute_stats(self) -> dict[str, Any]:
        df = self.df
        top_states = (df.groupby("State_Name")["Production"].sum()
                        .nlargest(10).reset_index()
                        .apply(lambda r: f"{r['State_Name']}: {int(r['Production']):,} tons", axis=1)
                        .tolist())

        top_crops = (df.groupby("Crop")["Production"].sum()
                       .nlargest(10).reset_index()
                       .apply(lambda r: f"{r['Crop']}: {int(r['Production']):,} tons", axis=1)
                       .tolist())

        top_yield = (df.groupby("Crop")["Yield"].mean().dropna()
                       .nlargest(10).reset_index()
                       .apply(lambda r: f"{r['Crop']}: {r['Yield']:.2f} t/unit", axis=1)
                       .tolist())

        season_prod = (df.groupby("Season")["Production"].sum()
                         .reset_index()
                         .apply(lambda r: f"{r['Season']}: {int(r['Production']):,} tons", axis=1)
                         .tolist())

        yearly = df.groupby("Crop_Year")["Production"].sum().sort_index()
        recent5 = yearly.tail(5)
        prev5   = yearly.iloc[-10:-5]
        trend_pct = round((recent5.mean() - prev5.mean()) / prev5.mean() * 100, 1) if len(prev5) and prev5.mean() else 0
        trend_dir = "up" if trend_pct > 0 else "down"

        state_crop = {}
        for s in df["State_Name"].unique():
            sub = df[df["State_Name"] == s]
            if len(sub):
                state_crop[s] = sub.groupby("Crop")["Production"].sum().idxmax()

        return {
            "total_records": len(df),
            "states": df["State_Name"].nunique(),
            "districts": df["District_Name"].nunique(),
            "crops": df["Crop"].nunique(),
            "years": f"{int(df['Crop_Year'].min())}–{int(df['Crop_Year'].max())}",
            "top_states": top_states,
            "top_crops": top_crops,
            "top_yield_crops": top_yield,
            "season_production": season_prod,
            "trend_pct": trend_pct,
            "trend_dir": trend_dir,
            "peak_year": int(yearly.idxmax()),
            "peak_production": int(yearly.max()),
            "total_production": int(df["Production"].sum()),
            "state_top_crop": state_crop,
            "cat_crops": df["cat_crop"].unique().tolist(),
            "seasons": df["Season"].unique().tolist(),
        }

    # ── System prompt with full dataset context ─────────────────
    def _build_system_prompt(self) -> str:
        s = self.stats
        state_crop_lines = "\n".join(
            f"  {st}: {cr}" for st, cr in list(s["state_crop"].items())[:15]
        ) if "state_crop" in s else "\n".join(
            f"  {st}: {cr}" for st, cr in list(s["state_top_crop"].items())[:15]
        )

        return f"""You are AgriVision AI, a senior agricultural data analyst and expert advisor. \
You have deep knowledge of Indian agriculture and access to a comprehensive crop production dataset.

## DATASET FACTS (use these to answer precisely)

- Total records: {s['total_records']:,}
- States: {s['states']} | Districts: {s['districts']} | Unique crops: {s['crops']}
- Years covered: {s['years']}
- Total national production: {s['total_production']:,} tons
- Production trend (last 5 vs prior 5 years): {s['trend_dir']} {abs(s['trend_pct'])}%
- Peak year: {s['peak_year']} ({s['peak_production']:,} tons)

### Top 10 States by Production:
{chr(10).join(s['top_states'])}

### Top 10 Crops by Production:
{chr(10).join(s['top_crops'])}

### Top 10 Crops by Average Yield:
{chr(10).join(s['top_yield_crops'])}

### Production by Season:
{chr(10).join(s['season_production'])}

### Top Crop per State (sample):
{state_crop_lines}

### Crop Categories: {', '.join(s['cat_crops'])}

## YOUR BEHAVIOUR
- Answer directly and confidently using the data above
- Give specific numbers, percentages, and comparisons
- When asked to compare states/crops, provide a clear ranked breakdown
- For trends, describe direction and magnitude
- Format responses with short paragraphs or bullet points — no walls of text
- If a question is outside the dataset scope, say so clearly and offer related insights
- Never repeat yourself or pad with filler phrases
- Keep answers concise but complete (3–8 sentences typically)
"""

    # ── Core chat via Anthropic API ──────────────────────────────
    def _request_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def chat(self, user_message: str, history: list[dict[str, Any]] | None = None) -> str:
        """
        Send message to Claude API with full conversation history.
        Returns assistant reply string.
        """
        messages = (history or []) + [{"role": "user", "content": user_message}]

        payload = {
            "model": self.model,
            "max_tokens": 600,
            "system": self.system_prompt,
            "messages": messages,
        }

        if not self.api_key:
            return self._fallback(user_message)

        try:
            resp = requests.post(
                self.ANTHROPIC_URL,
                headers=self._request_headers(),
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"].strip()
        except requests.RequestException:
            logger.warning("Anthropic request failed; using fallback response")
            return self._fallback(user_message)
        except Exception:
            logger.exception("Unexpected chatbot error")
            return self._fallback(user_message)

    # ── Intelligent fallback (when API unreachable) ──────────────
    def _fallback(self, q: str) -> str:
        s = self.stats
        q = q.lower()

        # State + crop combo
        for state, crop in s["state_top_crop"].items():
            if state.lower() in q and ("best" in q or "top" in q or "crop" in q):
                prod = int(self.df[
                    (self.df["State_Name"] == state) &
                    (self.df["Crop"] == crop)
                ]["Production"].sum())
                return (f"**{crop}** is the leading crop in **{state}**, "
                        f"with a total recorded production of **{prod:,} tons**. "
                        f"It consistently outperforms other crops in that region across seasons.")

        # Specific crop trend
        for crop in self.df["Crop"].unique():
            if crop.lower() in q and ("trend" in q or "year" in q or "production" in q):
                trend = (self.df[self.df["Crop"] == crop]
                           .groupby("Crop_Year")["Production"].sum()
                           .sort_index())
                lines = ", ".join(f"{yr}: {int(p):,}" for yr, p in trend.tail(5).items())
                pct = round((trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0] * 100, 1) if len(trend) > 1 else 0
                direction = "grown" if pct > 0 else "declined"
                return (f"**{crop}** production over the last 5 recorded years — {lines} tons. "
                        f"Overall it has **{direction} by {abs(pct)}%** across the full dataset period.")

        # Yield query
        if "yield" in q:
            top = s["top_yield_crops"][:5]
            return ("**Top 5 crops by average yield:**\n" +
                    "\n".join(f"- {t}" for t in top) +
                    "\n\nHigh-yield crops indicate efficient land use per unit area.")

        # Season query
        if "season" in q:
            lines = "\n".join(f"- {t}" for t in s["season_production"])
            return f"**Production breakdown by season:**\n{lines}"

        # State ranking
        if "state" in q and ("top" in q or "best" in q or "highest" in q):
            lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(s["top_states"][:5]))
            return f"**Top 5 producing states:**\n{lines}"

        # Trend
        if "trend" in q or "growth" in q or "increas" in q:
            return (f"National crop production has **{s['trend_dir']} by {abs(s['trend_pct'])}%** "
                    f"comparing the most recent 5 years to the prior 5 years. "
                    f"Peak production was recorded in **{s['peak_year']}** at **{s['peak_production']:,} tons**.")

        # Default
        return (f"I can answer questions about crop production across {s['states']} Indian states "
                f"from {s['years']}. Try asking about top states, best crops by yield, "
                f"seasonal trends, or production growth over time.")

    def get_status(self) -> dict:
        return {"model": self.model, "dataset_records": len(self.df), "api_enabled": bool(self.api_key)}

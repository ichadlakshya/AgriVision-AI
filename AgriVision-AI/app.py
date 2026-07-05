"""
AgriVision AI - Professional Agricultural Intelligence Platform
===============================================================
Enterprise-level agricultural analytics & production forecasting with AI Chatbot.
Modern, responsive design with professional theming.

Run with: streamlit run app.py
"""

import streamlit as st
import io
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_processor import DataProcessor
from insights_engine import InsightsEngine
from chatbot import AgriChatbot
from gps_map import GPSMapEngine
from predict import Predictor

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriVision AI - Agricultural Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# PROFESSIONAL STYLING & THEMING
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary-color: #2d8659;
        --secondary-color: #1e5a2e;
        --accent-color: #52b788;
        --dark-bg: #0f172a;
        --card-bg: #1a2332;
        --border-color: #2d4563;
        --text-primary: #e9ecef;
        --text-secondary: #a8b5c1;
    }

    * { margin: 0; padding: 0; }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1a2538 50%, #0f1f2e 100%);
        color: var(--text-primary);
    }

    /* Header Styling */
    .header-banner {
        background: linear-gradient(90deg, #2d8659 0%, #1e5a2e 50%, #0f5f2f 100%);
        padding: 30px 40px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(45, 134, 89, 0.15);
        border: 1px solid rgba(82, 183, 136, 0.2);
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.85);
        margin-top: 8px;
        font-weight: 400;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a2332 0%, #1f2d3f 100%);
        border: 1px solid rgba(82, 183, 136, 0.2);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2d8659, #52b788);
    }

    .kpi-card:hover {
        border-color: rgba(82, 183, 136, 0.5);
        box-shadow: 0 8px 32px rgba(45, 134, 89, 0.2);
        transform: translateY(-4px);
    }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #52b788;
        margin: 12px 0 8px 0;
        letter-spacing: -1px;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 8px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .kpi-icon {
        font-size: 2.5rem;
        margin-bottom: 8px;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--accent-color);
        margin: 30px 0 20px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Insight Boxes */
    .insight-box {
        background: rgba(26, 35, 50, 0.8);
        border-left: 4px solid var(--accent-color);
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 10px 0;
        font-size: 0.95rem;
        color: var(--text-primary);
        line-height: 1.6;
        border: 1px solid rgba(82, 183, 136, 0.1);
        border-left: 3px solid var(--accent-color);
    }

    .insight-box:hover {
        background: rgba(26, 35, 50, 1);
        border-color: rgba(82, 183, 136, 0.3);
    }

    /* Chat Interface */
    .chat-container {
        background: linear-gradient(135deg, #1a2332 0%, #1f2d3f 100%);
        border: 1px solid rgba(82, 183, 136, 0.2);
        border-radius: 12px;
        padding: 20px;
        height: 600px;
        overflow-y: auto;
        margin-bottom: 20px;
    }

    .chat-message-user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 12px;
    }

    .chat-bubble-user {
        background: linear-gradient(135deg, #2d8659 0%, #1e5a2e 100%);
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        color: #ffffff;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(45, 134, 89, 0.2);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .chat-message-bot {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 12px;
    }

    .chat-bubble-bot {
        background: rgba(82, 183, 136, 0.1);
        border: 1px solid rgba(82, 183, 136, 0.3);
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        color: var(--text-primary);
        max-width: 75%;
        word-wrap: break-word;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: rgba(26, 35, 50, 0.8) !important;
        border: 1px solid rgba(82, 183, 136, 0.2) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 2px rgba(82, 183, 136, 0.1) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2d8659 0%, #1e5a2e 100%) !important;
        color: white !important;
        border: 1px solid rgba(82, 183, 136, 0.3) !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #52b788 0%, #2d8659 100%) !important;
        box-shadow: 0 4px 16px rgba(45, 134, 89, 0.3) !important;
        transform: translateY(-2px) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(26, 35, 50, 0.5);
        border-bottom: 2px solid rgba(82, 183, 136, 0.2);
        border-radius: 0;
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-color) !important;
        border-bottom: 3px solid var(--accent-color) !important;
    }

    /* Sidebar */
    .stSidebar {
        background: linear-gradient(180deg, #0f1f2e 0%, #1a2538 100%);
        border-right: 1px solid rgba(82, 183, 136, 0.1);
    }

    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--accent-color);
        margin: 20px 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Divider */
    hr {
        border-color: rgba(82, 183, 136, 0.2);
        margin: 20px 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(26, 35, 50, 0.5);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(82, 183, 136, 0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(82, 183, 136, 0.7);
    }

    /* Expanders */
    .stExpander {
        background: rgba(26, 35, 50, 0.5);
        border: 1px solid rgba(82, 183, 136, 0.1);
        border-radius: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(82, 183, 136, 0.1);
    }

    /* Metric */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: var(--accent-color) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "crop_modified.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")


@st.cache_data(show_spinner="⚙️ Loading & processing data...")
def load_processed_data():
    processor = DataProcessor(DATA_PATH)
    df = processor.full_pipeline()
    return df, processor


@st.cache_resource(show_spinner="🤖 Loading ML model...")
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@st.cache_resource(show_spinner="🔮 Loading prediction engine...")
def load_predictor():
    try:
        return Predictor()
    except SystemExit:
        return None
    except Exception as exc:
        logger.warning("Prediction engine error: %s", exc)
        return None


df, processor = load_processed_data()
model_bundle = load_model()
predictor_engine = load_predictor()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 AgriVision AI")
    st.markdown("*Crop Production Intelligence Platform*")
    st.divider()

    states = ["All"] + sorted(df["State_Name"].unique().tolist())
    sel_state = st.selectbox("🗺️ State", states)

    filtered = df if sel_state == "All" else df[df["State_Name"] == sel_state]
    districts = ["All"] + sorted(filtered["District_Name"].unique().tolist())
    sel_district = st.selectbox("📍 District", districts)

    if sel_district != "All":
        filtered = filtered[filtered["District_Name"] == sel_district]

    crops = ["All"] + sorted(filtered["Crop"].unique().tolist())
    sel_crop = st.selectbox("🌱 Crop", crops)
    if sel_crop != "All":
        filtered = filtered[filtered["Crop"] == sel_crop]

    seasons = ["All"] + sorted(filtered["Season"].unique().tolist())
    sel_season = st.selectbox("🗓️ Season", seasons)
    if sel_season != "All":
        filtered = filtered[filtered["Season"] == sel_season]

    year_min, year_max = int(df["Crop_Year"].min()), int(df["Crop_Year"].max())
    sel_years = st.slider("📅 Year Range", year_min, year_max, (year_min, year_max))
    filtered = filtered[filtered["Crop_Year"].between(*sel_years)]

    st.divider()
    st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** records")

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("# 🌾 AgriVision AI")
st.markdown("#### Crop Production Forecasting & Agricultural Intelligence Platform")
st.divider()

# ─────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
total_prod = int(filtered["Production"].sum())
total_area = int(filtered["Area"].sum())
top_state = filtered.groupby("State_Name")["Production"].sum().idxmax() if len(filtered) else "N/A"
top_crop = filtered.groupby("Crop")["Production"].sum().idxmax() if len(filtered) else "N/A"

k1.metric("🌾 Total Production", f"{total_prod:,} T")
k2.metric("📐 Total Area", f"{total_area:,}")
k3.metric("🏆 Top State", top_state)
k4.metric("🥇 Top Crop", top_crop)

st.divider()

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🗺️ State Analysis", "🌱 Crop Analysis",
    "🔮 Forecast", "🧠 Explainable AI", "💡 AI Insights & Chat",
    "📡 GPS & Live Map"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📊 Dataset Overview")
    col1, col2 = st.columns(2)

    # Year-wise production trend
    with col1:
        yearly = filtered.groupby("Crop_Year")["Production"].sum().reset_index()
        fig = px.line(yearly, x="Crop_Year", y="Production",
                      title="📈 Year-wise Production Trend",
                      markers=True, color_discrete_sequence=["#52b788"])
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, width='stretch')

    # Season-wise production
    with col2:
        season_prod = filtered.groupby("Season")["Production"].sum().reset_index()
        fig2 = px.bar(season_prod, x="Season", y="Production",
                      title="🗓️ Season-wise Production",
                      color="Production", color_continuous_scale="Viridis")
        fig2.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig2, width='stretch')

    # Distribution plots
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(filtered, x="Production", nbins=60,
                            title="📉 Production Distribution (log scale)",
                            color_discrete_sequence=["#52b788"], log_y=True)
        fig3.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig3, width='stretch')

    with col4:
        cat_prod = filtered.groupby("cat_crop")["Production"].sum().reset_index()
        fig4 = px.pie(cat_prod, names="cat_crop", values="Production",
                      title="🥧 Production by Crop Category",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        fig4.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig4, width='stretch')

    # Correlation heatmap
    st.markdown("### 🔗 Correlation Analysis")
    num_cols = ["Area", "Production", "Yield", "Crop_Year",
                "State_Productivity_Score", "Seasonal_Productivity_Score"]
    corr = filtered[num_cols].corr().round(2)
    fig5 = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix",
                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig5.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig5, width='stretch')

# ══════════════════════════════════════════════════════════════════
# TAB 2 — STATE ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🗺️ State-wise Production Analysis")

    state_prod = filtered.groupby("State_Name")["Production"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(state_prod.head(20), x="Production", y="State_Name",
                 orientation="h", title="🏆 Top 20 States by Total Production",
                 color="Production", color_continuous_scale="Greens")
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        state_yield = filtered.groupby("State_Name")["Yield"].mean().sort_values(ascending=False).head(15).reset_index()
        fig2 = px.bar(state_yield, x="Yield", y="State_Name", orientation="h",
                      title="🌿 Top 15 States by Average Yield",
                      color="Yield", color_continuous_scale="Blues")
        fig2.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig2, width='stretch')

    with col2:
        state_area = filtered.groupby("State_Name")["Area"].sum().sort_values(ascending=False).head(15).reset_index()
        fig3 = px.bar(state_area, x="Area", y="State_Name", orientation="h",
                      title="📐 Top 15 States by Cultivated Area",
                      color="Area", color_continuous_scale="Oranges")
        fig3.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig3, width='stretch')

    # Choropleth map of India
    st.markdown("### 🗺️ India Crop Production Map")
    state_map = filtered.groupby("State_Name")["Production"].sum().reset_index()
    fig_map = px.choropleth(
        state_map,
        geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
        featureidkey="properties.ST_NM",
        locations="State_Name",
        color="Production",
        color_continuous_scale="YlGn",
        title="🌍 State-wise Crop Production (tons)",
        hover_data={"Production": ":,"},
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, width='stretch')

# ══════════════════════════════════════════════════════════════════
# TAB 3 — CROP ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🌱 Crop-wise Production Analysis")

    crop_prod = filtered.groupby("Crop")["Production"].sum().sort_values(ascending=False).head(20).reset_index()
    fig = px.bar(crop_prod, x="Crop", y="Production",
                 title="🥇 Top 20 Crops by Production",
                 color="Production", color_continuous_scale="Viridis")
    fig.update_layout(template="plotly_dark", height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        crop_yield = filtered.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head(15).reset_index()
        fig2 = px.bar(crop_yield, x="Crop", y="Yield",
                      title="🌿 Top Crops by Average Yield",
                      color="Yield", color_continuous_scale="Teal")
        fig2.update_layout(template="plotly_dark", height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig2, width='stretch')

    with col2:
        # Year-wise trend for selected crop
        if sel_crop != "All":
            crop_trend = filtered.groupby("Crop_Year")["Production"].sum().reset_index()
            fig3 = px.area(crop_trend, x="Crop_Year", y="Production",
                           title=f"📈 {sel_crop} Production Trend",
                           color_discrete_sequence=["#52b788"])
        else:
            cat_trend = filtered.groupby(["Crop_Year", "cat_crop"])["Production"].sum().reset_index()
            fig3 = px.line(cat_trend, x="Crop_Year", y="Production", color="cat_crop",
                           title="📈 Category-wise Production Trends")
        fig3.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig3, width='stretch')

    # Scatter: Area vs Production
    sample_scatter = filtered.sample(min(5000, len(filtered)), random_state=42)
    fig4 = px.scatter(sample_scatter, x="Area", y="Production", color="cat_crop",
                      title="📊 Area vs Production (sampled 5K points)",
                      hover_data=["State_Name", "Crop", "Crop_Year"],
                      opacity=0.6, color_discrete_sequence=px.colors.qualitative.Vivid)
    fig4.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig4, width='stretch')

    st.markdown("### 🔗 Relationship Analysis")
    rel_col1, rel_col2 = st.columns(2)

    with rel_col1:
        scatter_sample = filtered.sample(min(5000, len(filtered)), random_state=42)
        fig5 = px.scatter(
            scatter_sample,
            x="Area",
            y="Yield",
            color="Production",
            size=np.clip(scatter_sample["Yield"], 1, None),
            hover_data=["State_Name", "Crop", "Crop_Year", "Production"],
            title="📈 Area vs Yield Relationship",
            color_continuous_scale="Viridis",
            opacity=0.7,
        )
        fig5.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig5, width='stretch')

    with rel_col2:
        corr_cols = [
            col for col in [
                "Area", "Production", "Yield", "Crop_Year",
                "State_Productivity_Score", "District_Productivity_Score",
                "Crop_Popularity_Score", "Seasonal_Productivity_Score",
                "Historical_Avg_Production"
            ] if col in filtered.columns
        ]
        corr_matrix = filtered[corr_cols].corr().round(2)
        fig6 = px.imshow(
            corr_matrix,
            text_auto=True,
            title="🔗 Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig6.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig6, width='stretch')

    st.markdown("#### 🌾 State-Crop Production Heatmap")
    top_states = filtered.groupby("State_Name")["Production"].sum().nlargest(10).index
    top_crops = filtered.groupby("Crop")["Production"].sum().nlargest(10).index
    heatmap_df = filtered[
        filtered["State_Name"].isin(top_states) & filtered["Crop"].isin(top_crops)
    ].pivot_table(index="State_Name", columns="Crop", values="Production", aggfunc="sum", fill_value=0)
    if len(heatmap_df):
        fig7 = px.imshow(
            heatmap_df,
            text_auto=False,
            aspect="auto",
            title="State vs Crop Production Heatmap",
            color_continuous_scale="YlGn",
        )
        fig7.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig7, width='stretch')

# ══════════════════════════════════════════════════════════════════
# TAB 4 — PRODUCTION FORECAST
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔮 Crop Production Forecast")

    if predictor_engine is None:
        st.warning("⚠️ No trained model found. Run `python train.py` first to train and save the model.")
        st.info("The training script will train 6 ML models, select the best, and save it to `models/best_model.joblib`.")
    else:
        prediction_df = predictor_engine.df
        st.success("✅ Using the standalone prediction engine from predict.py")
        st.markdown("---")
        st.markdown("#### 🎛️ Configure Prediction")

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            pred_state = st.selectbox("State", sorted(prediction_df["State_Name"].unique()), key="forecast_state")
            pred_district = st.selectbox(
                "District",
                sorted(prediction_df[prediction_df["State_Name"] == pred_state]["District_Name"].unique()),
                key="forecast_district",
            )
        with pc2:
            pred_crop = st.selectbox("Crop", sorted(prediction_df["Crop"].unique()), key="forecast_crop")
            pred_season = st.selectbox("Season", sorted(prediction_df["Season"].unique()), key="forecast_season")
        with pc3:
            pred_year = st.number_input("Crop Year", 1997, 2035, 2016, key="forecast_year")
            pred_area = st.number_input("Area (hectares)", 1, 10_000_000, 1000, key="forecast_area")

        if st.button("🚀 Predict Production", width='stretch'):
            result = predictor_engine.predict_one(
                pred_state,
                pred_district,
                pred_crop,
                pred_season,
                int(pred_year),
                float(pred_area),
            )

            st.markdown("---")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("🌾 Predicted Production", f"{result['Predicted_Production']:,} tons")
            r2.metric("📐 Area", f"{result['Area']:,.0f} ha")
            r3.metric("🌿 Yield", f"{result['Yield_t_per_ha']:.2f} t/ha")
            r4.metric("📊 Confidence Band", f"{result['Confidence_Low']:,} - {result['Confidence_High']:,}")

            if result["Historical_Avg"]:
                diff_pct = result["vs_Historical_pct"] or 0
                symbol = "🔼" if diff_pct > 0 else "🔽"
                st.info(
                    f"{symbol} Prediction is **{abs(diff_pct)}%** {'above' if diff_pct > 0 else 'below'} "
                    f"the historical average of {int(result['Historical_Avg']):,} tons for {pred_crop} in {pred_state}."
                )

            st.json(result)

        st.markdown("---")
        st.markdown("#### 📁 Batch Prediction")
        batch_upload = st.file_uploader(
            "Upload a CSV with State_Name, District_Name, Crop, Season, Crop_Year, Area",
            type=["csv"],
            key="batch_prediction_upload",
        )

        if batch_upload is not None:
            batch_df = pd.read_csv(io.BytesIO(batch_upload.getvalue()))
            st.dataframe(batch_df.head(20), width='stretch')

            if st.button("Run Batch Prediction", width='stretch', key="run_batch_prediction"):
                try:
                    batch_result = predictor_engine.predict_dataframe(batch_df)
                    st.success(f"Completed {len(batch_result):,} predictions.")
                    st.dataframe(batch_result, width='stretch')
                    st.download_button(
                        "Download Predictions CSV",
                        batch_result.to_csv(index=False).encode("utf-8"),
                        file_name="agrivision_predictions.csv",
                        mime="text/csv",
                        width='stretch',
                    )
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")

# ══════════════════════════════════════════════════════════════════
# TAB 5 — EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🧠 Explainable AI — SHAP Analysis")

    if model_bundle is None:
        st.warning("⚠️ Train the model first using `python train.py`.")
    else:
        shap_img = os.path.join(os.path.dirname(__file__), "reports", "shap_importance.png")
        shap_summary = os.path.join(os.path.dirname(__file__), "reports", "shap_summary.png")

        if os.path.exists(shap_img):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Feature Importance (SHAP)")
                st.image(shap_img, width='stretch')
            if os.path.exists(shap_summary):
                with col2:
                    st.markdown("#### SHAP Summary Plot")
                    st.image(shap_summary, width='stretch')
        else:
            st.info("SHAP plots will appear here after running `python train.py`.")

        st.markdown("---")
        st.markdown("#### 🔎 Understanding the Prediction")
        st.markdown("""
        **How SHAP works in AgriVision AI:**

        Each prediction can be decomposed into feature contributions:

        ```
        Prediction = Base Value + Σ(SHAP contributions)
        ```

        **Example:**
        ```
        Predicted Production: 12,450 tons
        ─────────────────────────────────
        Base Value:              3,200 tons
        + Area:                 +4,500 tons
        + Crop (Wheat):         +2,800 tons
        + State (Punjab):       +2,100 tons
        + Season (Rabi):        +1,200 tons
        - Yield (low):          -1,350 tons
        ```

        This means the area planted had the strongest positive impact,
        while below-average historical yield pulled the estimate down.
        """)

# ══════════════════════════════════════════════════════════════════
# TAB 6 — AI INSIGHTS & CHATBOT
# ══════════════════════════════════════════════════════════════════
with tab6:
    ins_col, chat_col = st.columns([1, 1])

    with ins_col:
        st.markdown("### 💡 Auto-Generated Insights")
        engine = InsightsEngine(filtered if len(filtered) > 100 else df)
        all_insights = engine.generate_all()

        for category, insights in all_insights.items():
            with st.expander(f"📌 {category}", expanded=(category == "Top Producing States")):
                for ins in insights:
                    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    with chat_col:
        st.markdown("### 🤖 AgriVision AI Chatbot")
        st.caption("Powered by Claude (Anthropic) · Knows your full dataset")

        @st.cache_resource
        def get_chatbot():
            return AgriChatbot(df)

        bot = get_chatbot()
        st.success(f"🟢 Claude AI connected · {len(df):,} records loaded as context")

        # ── Session state init ────────────────────────────────────
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_history" not in st.session_state:
            # API-format history (role/content pairs, no system message)
            st.session_state.chat_history = []

        # ── Render existing messages ──────────────────────────────
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🌾"):
                st.markdown(msg["content"])

        # ── Quick-prompt buttons ──────────────────────────────────
        if not st.session_state.messages:
            st.markdown("**Try asking:**")
            btn_cols = st.columns(2)
            quick_prompts = [
                ("🏆 Top producing states", "Which states have the highest crop production? Give me a ranked breakdown with percentages."),
                ("🌿 Best crops by yield", "Which crops have the highest yield per unit area? Explain why."),
                ("📈 Production trends", "How has national crop production trended over the years? When was the peak?"),
                ("🌾 Punjab analysis",  "Give me a detailed production analysis for Punjab — top crops, growth trend, and seasonal breakdown."),
                ("🗓️ Season comparison", "Compare Kharif vs Rabi season production. Which dominates and why?"),
                ("🔮 Forecast insight",  "Which crops and states show the most growth potential based on historical trends?"),
            ]
            for i, (label, prompt) in enumerate(quick_prompts):
                col = btn_cols[i % 2]
                if col.button(label, key=f"qp_{i}", width='stretch'):
                    st.session_state._quick_prompt = prompt
                    st.rerun()

        # ── Handle quick prompt click ─────────────────────────────
        if hasattr(st.session_state, "_quick_prompt") and st.session_state._quick_prompt:
            user_msg = st.session_state._quick_prompt
            st.session_state._quick_prompt = None

            st.session_state.messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_msg)

            with st.chat_message("assistant", avatar="🌾"):
                with st.spinner("Analysing data..."):
                    reply = bot.chat(user_msg, history=st.session_state.chat_history)
                st.markdown(reply)

            st.session_state.chat_history.append({"role": "user",      "content": user_msg})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

        # ── Main chat input ───────────────────────────────────────
        user_input = st.chat_input("Ask about crops, states, yields, trends…")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="🌾"):
                with st.spinner("Thinking…"):
                    reply = bot.chat(user_input, history=st.session_state.chat_history)
                st.markdown(reply)

            st.session_state.chat_history.append({"role": "user",      "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

        # ── Clear chat button ─────────────────────────────────────
        if st.session_state.messages:
            if st.button("🗑️ Clear chat", key="clear_chat"):
                st.session_state.messages     = []
                st.session_state.chat_history = []
                st.rerun()

# Footer
st.divider()
st.markdown(
    "<center style='color:#6c757d;font-size:0.8rem;'>"
    "AgriVision AI | Built with Streamlit + Plotly + XGBoost + SHAP | "
    "Data: India Crop Production 1997–2015"
    "</center>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════
# TAB 7 — GPS & LIVE MAP
# ══════════════════════════════════════════════════════════════════
with tab7:
    try:
        from streamlit_folium import st_folium
        folium_available = True
    except ImportError:
        folium_available = False

    st.markdown("### 📡 GPS & Real-Time Agricultural Map")

    if not folium_available:
        st.error("Install `streamlit-folium`: `pip install streamlit-folium`")
        st.stop()

    @st.cache_resource
    def get_gps_engine():
        return GPSMapEngine(df)

    gps = get_gps_engine()

    # ── GPS Input section ─────────────────────────────────────────
    gps_col, ctrl_col = st.columns([2, 1])

    with ctrl_col:
        st.markdown("#### 📍 Your Location")

        # Manual or auto GPS
        gps_mode = st.radio(
            "Location input", ["🌐 Auto-detect (browser)", "✏️ Manual entry"],
            horizontal=True
        )

        user_lat, user_lon = None, None

        if gps_mode == "✏️ Manual entry":
            user_lat = st.number_input("Latitude",  value=28.6139, format="%.4f")
            user_lon = st.number_input("Longitude", value=77.2090, format="%.4f")
        else:
            st.info("Click **📍 Find my location** button on the map (top-right corner).")
            # JS component to grab browser GPS
            st.html("""
            <div id="gps-status" style="color:#aaa;font-size:13px;">Waiting for location…</div>
            <script>
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(pos) {
                    const lat = pos.coords.latitude.toFixed(5);
                    const lon = pos.coords.longitude.toFixed(5);
                    document.getElementById('gps-status').innerHTML =
                        '✅ GPS acquired: ' + lat + ', ' + lon;
                    // Post to Streamlit session via URL param trick
                    window.parent.postMessage({type:'gps', lat:lat, lon:lon}, '*');
                }, function(err) {
                    document.getElementById('gps-status').innerHTML =
                        '⚠️ ' + err.message + ' — use manual entry.';
                });
            }
            </script>
            """)
            # Fallback coords (New Delhi) when browser GPS not available
            user_lat = st.session_state.get("gps_lat", 28.6139)
            user_lon = st.session_state.get("gps_lon", 77.2090)

        # Map filters
        st.markdown("#### 🎛️ Map Filters")
        map_crop   = st.selectbox("Crop filter",   ["All"] + sorted(df["Crop"].unique().tolist()),   key="map_crop")
        map_season = st.selectbox("Season filter", ["All"] + sorted(df["Season"].unique().tolist()), key="map_season")
        map_mode   = st.radio("Map mode", ["🌾 Production Map", "✏️ Field Mapper"], horizontal=True)

    with gps_col:
        if map_mode == "🌾 Production Map":
            fmap = gps.build_india_production_map(
                user_lat=user_lat,
                user_lon=user_lon,
                selected_crop=map_crop,
                selected_season=map_season,
            )
            map_data = st_folium(fmap, width=700, height=520, returned_objects=["last_clicked"])

            # Capture clicked point
            if map_data and map_data.get("last_clicked"):
                click_lat = map_data["last_clicked"]["lat"]
                click_lon = map_data["last_clicked"]["lng"]
                st.session_state["clicked_lat"] = click_lat
                st.session_state["clicked_lon"] = click_lon

        else:
            center_lat = user_lat or 20.5937
            center_lon = user_lon or 78.9629
            fmap = gps.build_field_mapper(center_lat, center_lon)
            st_folium(fmap, width=700, height=520)
            st.caption("✏️ Use polygon/rectangle tools to draw your field. Area shows automatically.")

    # ── Location intelligence panel ───────────────────────────────
    st.divider()
    loc_lat = st.session_state.get("clicked_lat", user_lat)
    loc_lon = st.session_state.get("clicked_lon", user_lon)

    if loc_lat and loc_lon:
        nearest = gps.nearest_state(loc_lat, loc_lon)
        recs    = gps.recommend_crops_for_location(loc_lat, loc_lon, season=map_season)

        info_col, rec_col = st.columns(2)

        with info_col:
            st.markdown("#### 📍 Location Intelligence")
            st.metric("Nearest State",       nearest["state"])
            st.metric("Distance to centroid", f"{nearest['distance_km']} km")
            st.metric("Top Crop in Region",  nearest["top_crop"])
            st.metric("State Total Production", f"{nearest['total_production']:,} tons")

            st.markdown("#### 🔮 Quick Predict from GPS")
            gps_area = st.number_input("Enter your field area (ha)", min_value=1, value=500, key="gps_area")
            if st.button("⚡ Predict Production for this location", width='stretch'):
                if model_bundle:
                    import numpy as _np
                    model = model_bundle["model"]
                    feature_names = model_bundle["feature_names"]
                    state_rows = df[df["State_Name"] == nearest["state"]]
                    if len(state_rows):
                        row = state_rows.head(1)[feature_names].copy()
                        row["Area"]    = gps_area
                        row["Log_Area"] = _np.log1p(gps_area)
                        pred = max(0, float(model.predict(row.fillna(0))[0]))
                        st.success(f"🌾 Predicted: **{int(pred):,} tons** for {gps_area:,} ha in {nearest['state']}")
                        st.caption(f"Top crop: {nearest['top_crop']} | Yield: {round(pred/gps_area,2)} t/ha")
                else:
                    st.warning("Train the model first: `python train.py`")

        with rec_col:
            st.markdown(f"#### 🌱 Top Crops for {nearest['state']}")
            if len(recs):
                recs_display = recs.rename(columns={
                    "Crop": "Crop",
                    "Avg_Yield": "Avg Yield (t/unit)",
                    "Total_Production": "Total Production",
                    "Seasons": "Best Seasons",
                })
                st.dataframe(
                    recs_display[["Crop", "Avg Yield (t/unit)", "Total Production", "Best Seasons"]],
                    width='stretch',
                    hide_index=True,
                )

                # Mini bar chart of yields
                fig_rec = px.bar(
                    recs, x="Crop", y="Avg_Yield",
                    title=f"Recommended Crops by Yield — {nearest['state']}",
                    color="Avg_Yield", color_continuous_scale="Greens",
                )
                fig_rec.update_layout(template="plotly_dark", height=280)
                st.plotly_chart(fig_rec, width='stretch')
            else:
                st.info("No crop data for this region.")

    # ── Live location tracking hint ───────────────────────────────
    st.divider()
    st.markdown("#### ℹ️ How GPS Works in AgriVision AI")
    c1, c2, c3 = st.columns(3)
    c1.info("**🌐 Browser GPS**\nClick the locate button on the map. Your browser requests GPS permission and pins your position.")
    c2.info("**📍 Click-to-analyse**\nClick anywhere on the production map to instantly see the nearest state's crop statistics and recommendations.")
    c3.info("**✏️ Field Mapper**\nSwitch to Field Mapper mode, draw your field boundary, and the tool calculates area for production forecasting.")

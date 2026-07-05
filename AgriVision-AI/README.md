# 🌾 AgriVision AI
### Crop Production Forecasting & Agricultural Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> An agricultural intelligence platform for predicting, analyzing, and explaining crop production across Indian states using machine learning, explainable AI, and a configurable chatbot.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Interactive Dashboard** | 6-tab Streamlit dashboard with Plotly charts |
| 🔮 **ML Forecasting** | XGBoost/LightGBM/Random Forest production prediction |
| 🧠 **Explainable AI** | SHAP feature importance & waterfall plots |
| 🗺️ **India Map** | Choropleth state-wise production visualization |
| 💡 **Auto Insights** | Dynamically generated agricultural intelligence |
| 🤖 **AI Chatbot** | Anthropic-powered chatbot with rule-based fallback |
| 📐 **Feature Engineering** | Yield, productivity scores, historical averages |

---

## 🗂️ Project Structure

```
AgriVision-AI/
├── data/
│   └── crop_modified.csv        # Raw dataset
├── models/
│   └── best_model.joblib        # Saved best ML model
├── reports/
│   ├── shap_importance.png      # SHAP feature importance plot
│   └── shap_summary.png         # SHAP beeswarm summary plot
├── src/
│   ├── data_processor.py        # Data loading & feature engineering
│   ├── model_trainer.py         # ML training & evaluation
│   ├── explainability.py        # SHAP explainability engine
│   ├── insights_engine.py       # Auto insight generation
│   ├── chatbot.py               # Anthropic chatbot + fallback
│   ├── nlp_chatbot.py           # Dataset-aware TF-IDF chatbot
│   └── runtime.py               # Env + logging helpers
├── static/
├── templates/
├── app.py                       # Main Streamlit dashboard
├── ui_app.py                    # Production Flask app
├── predict.py                   # Prediction CLI
├── train.py                     # Model training script
├── Dockerfile
├── Procfile
├── wsgi.py
├── gunicorn.conf.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/AgriVision-AI.git
cd AgriVision-AI
pip install -r requirements.txt
copy .env.example .env
```

### 2. Configure Environment

Set your production values in `.env`:

```bash
FLASK_SECRET_KEY=your-secret-key
FLASK_DEBUG=false
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### 3. Train the Model

```bash
python train.py
```

This trains 6 ML models, picks the best, saves it, and generates SHAP plots.

### 4. Launch the Flask App

```bash
python ui_app.py
```

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

### 6. Optional: Enable Anthropic Chatbot Access

```bash
# Set ANTHROPIC_API_KEY in .env
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | India Crop Production Statistics |
| Records | 242,361 |
| States | 33 |
| Crops | 124 |
| Years | 1997–2015 |
| Target | Production (tons) |

**Columns:** `State_Name`, `District_Name`, `Crop_Year`, `Season`, `Crop`, `Area`, `Production`, `cat_crop`

---

## 🤖 ML Models & Results

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| **XGBoost** ✅ | **0.9368** | best | best |
| LightGBM | 0.8509 | — | — |
| Random Forest | ~0.88 | — | — |
| Decision Tree | ~0.82 | — | — |
| Linear Regression | ~0.45 | — | — |
| CatBoost | ~0.87 | — | — |

---

## 🧠 Feature Engineering

| Feature | Description |
|---|---|
| `Yield` | Production / Area |
| `Decade` | Decade bin of Crop_Year |
| `State_Productivity_Score` | Mean production per state |
| `District_Productivity_Score` | Mean production per district |
| `Crop_Popularity_Score` | Frequency of crop in dataset |
| `Seasonal_Productivity_Score` | Mean production per season |
| `Historical_Avg_Production` | Mean production per (State, Crop) |
| `Log_Area` | Log-transformed area |

---

## 💡 Sample Auto-Insights

- 🌾 **Uttar Pradesh** contributed **18.2%** of national production.
- 📈 National crop production **increased by 14.3%** comparing last 5 vs prior 5 years.
- 🌱 **Sugarcane** has the highest average yield: **52.4 tons/unit area**.
- 🗓️ **Kharif** season accounts for **52.1%** of total production.

## 📸 Screenshots

Add exported screenshots to a `docs/screenshots/` folder and reference them here for a portfolio-ready presentation.

- Home dashboard
- Analytics dashboard
- Chatbot conversation view
- Prediction result card

## 🔌 API Reference

The Flask app exposes these endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home page with prediction form and stats |
| `GET` | `/chatbot` | Dedicated chatbot UI |
| `GET` | `/analytics` | Analytics dashboard |
| `POST` | `/api/chat` | Chatbot response for a user message |
| `POST` | `/api/predict` | Single crop production prediction |
| `GET` | `/api/analytics/charts` | Chart-ready analytics payload |
| `GET` | `/api/insights` | Auto-generated insights |
| `GET` | `/api/stats` | Platform summary statistics |

Example prediction payload:

```json
{
	"state": "Punjab",
	"district": "LUDHIANA",
	"crop": "Wheat",
	"season": "Rabi",
	"year": 2020,
	"area": 5000
}
```

## 🚢 Deployment

### Docker

```bash
docker build -t agrivision-ai .
docker run -p 5000:5000 --env-file .env agrivision-ai
```

### Gunicorn / Procfile

Production WSGI entrypoint: `wsgi.py`

```bash
gunicorn -c gunicorn.conf.py wsgi:app
```

### GitHub Actions

The workflow in `.github/workflows/ci.yml` installs dependencies and runs a Python syntax check on push and pull requests.

## 🔐 Security Notes

- Keep secrets in `.env` or your cloud secret manager.
- Restrict `CORS_ORIGINS` to trusted hosts.
- Leave `FLASK_DEBUG=false` in production.
- Use HTTPS behind a reverse proxy.
- Rotate third-party API keys periodically.

## ✅ Deployment Checklist

- [ ] Copy `.env.example` to `.env`
- [ ] Set `FLASK_SECRET_KEY`
- [ ] Set `ANTHROPIC_API_KEY` if chatbot API access is needed
- [ ] Train the model with `python train.py`
- [ ] Confirm `models/best_model.joblib` exists
- [ ] Start the Flask app with `python ui_app.py`
- [ ] Test `/api/chat`, `/api/predict`, `/api/analytics/charts`, and `/api/insights`
- [ ] Build the Docker image successfully
- [ ] Deploy with Gunicorn or another WSGI server

---

## 🔮 Future Scope

- [ ] Weather API integration for real-time forecasting
- [ ] District-level choropleth maps
- [ ] Time-series LSTM forecasting
- [ ] Crop price prediction module
- [ ] Mobile app with React Native
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Satellite imagery integration (NDVI)

---

## 📄 License

MIT License — free for personal, academic, and commercial use.

---

## 👤 Author

Built as a portfolio project demonstrating end-to-end ML engineering, data science, and full-stack AI development.

"""AgriVision AI Flask application."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processor import DataProcessor
from insights_engine import InsightsEngine
from nlp_chatbot import NLPChatbot
from predict import Predictor
from runtime import configure_logging, env, env_bool, env_list

# ─────────────────────────────────────────────────────────────────
# FLASK APP INITIALIZATION
# ─────────────────────────────────────────────────────────────────
logger = configure_logging(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = env("FLASK_SECRET_KEY", "dev-only-change-me")
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = int(env("MAX_CONTENT_LENGTH", "1048576") or "1048576")

allowed_origins = env_list("CORS_ORIGINS", ["http://localhost:5000", "http://127.0.0.1:5000"])
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

def initialize_app() -> None:
    """Load shared dataset-backed services once per process."""

    if app.config.get("initialized") or app.config.get("initialization_attempted"):
        return

    app.config["initialization_attempted"] = True

    try:
        data_path = BASE_DIR / "data" / "crop_modified.csv"
        processor = DataProcessor(str(data_path))
        df = processor.full_pipeline()

        app.config.update(
            {
                "df": df,
                "processor": processor,
                "chatbot": NLPChatbot(df),
                "insights": InsightsEngine(df),
                "predictor": Predictor(),
                "initialized": True,
            }
        )
        logger.info("Initialized AgriVision AI with %s records", len(df))
    except Exception:
        logger.exception("Initialization error")
        app.config["initialization_error"] = (
            "AgriVision AI is starting up and cannot load the dataset or prediction engine yet."
        )


@app.before_request
def ensure_initialized() -> None:
    """Initialize the app lazily before the first request."""

    initialize_app()


def _dataset_missing_response():
    message = app.config.get("initialization_error", "Service unavailable.")
    return jsonify({"status": "error", "error": message}), 503


def _dataset_missing_template(template_name: str, **context):
    context.setdefault("error", app.config.get("initialization_error", "Service unavailable."))
    if template_name == 'index.html':
        context.setdefault(
            'stats',
            {
                'total_records': 0,
                'states': 0,
                'crops': 0,
                'districts': 0,
                'years': 'N/A',
                'total_production': 0,
            },
        )
        context.setdefault(
            'options',
            {
                'states': [],
                'districts': [],
                'crops': [],
                'seasons': [],
                'default_year': 0,
                'district_map': {},
                'crop_map': {},
            },
        )
    return render_template(template_name, **context), 503


def _build_home_context(df):
    stats = {
        "total_records": len(df),
        "states": df["State_Name"].nunique(),
        "crops": df["Crop"].nunique(),
        "districts": df["District_Name"].nunique(),
        "years": f"{int(df['Crop_Year'].min())}-{int(df['Crop_Year'].max())}",
        "total_production": int(df["Production"].sum()),
    }
    options = {
        "states": sorted(df["State_Name"].dropna().unique().tolist()),
        "districts": sorted(df["District_Name"].dropna().unique().tolist()),
        "crops": sorted(df["Crop"].dropna().unique().tolist()),
        "seasons": sorted(df["Season"].dropna().unique().tolist()),
        "default_year": int(df["Crop_Year"].max()),
        "district_map": {
            state: sorted(group["District_Name"].dropna().unique().tolist())
            for state, group in df.groupby("State_Name")
        },
        "crop_map": {
            state: sorted(group["Crop"].dropna().unique().tolist())
            for state, group in df.groupby("State_Name")
        },
    }
    return stats, options

# ─────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    """Home page with dashboard overview"""
    try:
        df = app.config.get('df')
        if df is None:
            return _dataset_missing_template('index.html', stats={}, options={})

        stats, options = _build_home_context(df)
        return render_template('index.html', stats=stats, options=options)
    except Exception:
        logger.exception('Failed to render home page')
        return render_template('index.html', stats={}, options={}, error='Unable to load dashboard data.'), 500

# ─────────────────────────────────────────────────────────────────
# CHATBOT INTERFACE
# ─────────────────────────────────────────────────────────────────
@app.route('/chatbot')
def chatbot():
    """Dedicated chatbot interface"""
    if not app.config.get('initialized'):
        return _dataset_missing_template('chatbot.html')
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat API endpoint"""
    try:
        data = request.json
        if data is None:
            return jsonify({'error': 'Please send a JSON request body.'}), 400

        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Please enter a question about agriculture.'}), 400
        
        # Get chatbot instance
        chatbot = app.config.get('chatbot')
        if not chatbot:
            return _dataset_missing_response()
        
        # Generate response using NLP matching
        response = chatbot.get_response(message)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
    except Exception:
        logger.exception('Chat endpoint failed')
        return jsonify({
            'error': 'Unable to process your message right now. Please try again in a moment.',
            'status': 'error'
        }), 500

# ─────────────────────────────────────────────────────────────────
# ANALYTICS & INSIGHTS
# ─────────────────────────────────────────────────────────────────
@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    try:
        df = app.config.get('df')
        if df is None:
            return _dataset_missing_template('analytics.html')

        recent_year = int(df['Crop_Year'].max()) - 4
        recent_df = df[df['Crop_Year'] >= recent_year]

        # Full-range aggregates
        top_crops = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).to_dict()
        top_states = df.groupby('State_Name')['Production'].sum().sort_values(ascending=False).to_dict()
        seasonal = df.groupby('Season')['Production'].sum().to_dict()
        yearly = df.groupby('Crop_Year')['Production'].sum().to_dict()
        yield_by_crop = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).to_dict()
        top_districts = df.groupby('District_Name')['Production'].sum().sort_values(ascending=False).to_dict()

        # Recent-year aggregates for the "Recent Years" filter
        recent_top_crops = recent_df.groupby('Crop')['Production'].sum().sort_values(ascending=False).to_dict()
        recent_top_states = recent_df.groupby('State_Name')['Production'].sum().sort_values(ascending=False).to_dict()
        recent_seasonal = recent_df.groupby('Season')['Production'].sum().to_dict()
        recent_yearly = recent_df.groupby('Crop_Year')['Production'].sum().to_dict()
        recent_yield_by_crop = recent_df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).to_dict()
        recent_top_districts = recent_df.groupby('District_Name')['Production'].sum().sort_values(ascending=False).to_dict()
        
        return render_template('analytics.html', 
                             top_crops=json.dumps(top_crops),
                             top_states=json.dumps(top_states),
                             seasonal=json.dumps(seasonal),
                             yearly=json.dumps(yearly),
                             yield_by_crop=json.dumps(yield_by_crop),
                             top_districts=json.dumps(top_districts),
                             recent_year=recent_year,
                             recent_top_crops=json.dumps(recent_top_crops),
                             recent_top_states=json.dumps(recent_top_states),
                             recent_seasonal=json.dumps(recent_seasonal),
                             recent_yearly=json.dumps(recent_yearly),
                             recent_yield_by_crop=json.dumps(recent_yield_by_crop),
                             recent_top_districts=json.dumps(recent_top_districts))
    except Exception:
        logger.exception('Failed to render analytics page')
        return render_template('analytics.html', error='Unable to load analytics data right now.'), 500

@app.route('/api/analytics/charts')
def get_charts():
    """Get chart data for analytics"""
    try:
        df = app.config.get('df')
        if df is None:
            return _dataset_missing_response()

        recent_year = int(df['Crop_Year'].max()) - 4
        recent_df = df[df['Crop_Year'] >= recent_year]
        
        charts = {
            'top_crops': df.groupby('Crop')['Production'].sum().sort_values(ascending=False).to_dict(),
            'top_states': df.groupby('State_Name')['Production'].sum().sort_values(ascending=False).to_dict(),
            'seasonal': df.groupby('Season')['Production'].sum().to_dict(),
            'yearly': df.groupby('Crop_Year')['Production'].sum().to_dict(),
            'yield_by_crop': df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).to_dict(),
            'top_districts': df.groupby('District_Name')['Production'].sum().sort_values(ascending=False).to_dict(),
            'recent_year': recent_year,
            'recent_top_crops': recent_df.groupby('Crop')['Production'].sum().sort_values(ascending=False).to_dict(),
            'recent_top_states': recent_df.groupby('State_Name')['Production'].sum().sort_values(ascending=False).to_dict(),
            'recent_seasonal': recent_df.groupby('Season')['Production'].sum().to_dict(),
            'recent_yearly': recent_df.groupby('Crop_Year')['Production'].sum().to_dict(),
            'recent_yield_by_crop': recent_df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).to_dict(),
            'recent_top_districts': recent_df.groupby('District_Name')['Production'].sum().sort_values(ascending=False).to_dict(),
        }
        
        return jsonify(charts)
    except Exception:
        logger.exception('Failed to build chart data')
        return jsonify({'error': 'Unable to build analytics charts right now.'}), 500

# ─────────────────────────────────────────────────────────────────
# INSIGHTS
# ─────────────────────────────────────────────────────────────────
@app.route('/api/insights')
def get_insights():
    """Generate insights"""
    try:
        insights = app.config.get('insights')
        if not insights:
            return _dataset_missing_response()
        
        # src/insights_engine.py exposes generate_all() (categories -> lists)
        result = insights.generate_all()
        return jsonify(result)
    except Exception:
        logger.exception('Failed to generate insights')
        return jsonify({'error': 'Unable to generate insights right now.'}), 500

# ─────────────────────────────────────────────────────────────────
# PREDICTION API
# ─────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    """Make production prediction"""
    try:
        data = request.get_json(silent=True) or {}
        predictor = app.config.get('predictor')
        if predictor is None:
            return jsonify({'error': 'Prediction engine is not ready yet.'}), 503

        required_fields = ['state', 'district', 'crop', 'season', 'year', 'area']
        missing_fields = [field for field in required_fields if field not in data or data[field] in (None, '')]
        if missing_fields:
            return jsonify({'error': f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        result = predictor.predict_one(
            state=str(data.get('state')),
            district=str(data.get('district')),
            crop=str(data.get('crop')),
            season=str(data.get('season')),
            year=int(data.get('year')),
            area=float(data.get('area')),
        )

        return jsonify({
            'prediction': float(result['Predicted_Production']),
            'yield': float(result['Yield_t_per_ha']),
            'historical_avg': result['Historical_Avg'],
            'confidence_low': result['Confidence_Low'],
            'confidence_high': result['Confidence_High'],
            'details': result,
            'status': 'success'
        })
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        logger.exception('Prediction endpoint failed')
        return jsonify({'error': 'Unable to generate a prediction right now.'}), 500

# ─────────────────────────────────────────────────────────────────
# ABOUT & INFO
# ─────────────────────────────────────────────────────────────────
@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/stats')
def get_stats():
    """Get platform statistics"""
    try:
        df = app.config.get('df')
        if df is None:
            return _dataset_missing_response()
        
        stats = {
            'total_records': len(df),
            'states': df['State_Name'].nunique(),
            'crops': df['Crop'].nunique(),
            'districts': df['District_Name'].nunique(),
            'years_covered': f"{int(df['Crop_Year'].min())}-{int(df['Crop_Year'].max())}",
            'total_production': int(df['Production'].sum()),
            'avg_yield': round(df['Yield'].mean(), 2),
        }
        
        return jsonify(stats)
    except Exception:
        logger.exception('Failed to build platform stats')
        return jsonify({'error': 'Unable to load platform statistics right now.'}), 500

# ─────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

# ─────────────────────────────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    initialize_app()
    app.run(
        host=env('FLASK_HOST', '0.0.0.0'),
        port=int(env('FLASK_PORT', '5000') or '5000'),
        debug=env_bool('FLASK_DEBUG', False),
        threaded=True,
    )

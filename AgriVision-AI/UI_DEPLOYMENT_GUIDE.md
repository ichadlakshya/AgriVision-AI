# AgriVision AI - Professional Web UI

A modern, professional web interface for deploying the AgriVision AI chatbot and analytics platform.

## 📋 Overview

This web UI provides a sleek, enterprise-grade interface for:
- 🤖 **AI Chatbot** - Interactive agricultural intelligence assistant
- 📊 **Analytics Dashboard** - Real-time data visualizations
- 💡 **Insights Engine** - Automated agricultural intelligence
- 🌾 **Professional Design** - Modern green agricultural theme

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the Web Application**
```bash
python ui_app.py
```

3. **Access the Application**
Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure

```
AgriVision-AI/
├── ui_app.py                 # Flask application entry point
├── templates/                # HTML templates
│   ├── base.html            # Base template with navigation
│   ├── index.html           # Home page
│   ├── chatbot.html         # Chatbot interface
│   ├── analytics.html       # Analytics dashboard
│   ├── about.html           # About page
│   ├── 404.html             # 404 error page
│   └── 500.html             # 500 error page
├── static/
│   ├── css/
│   │   └── style.css        # Professional styling
│   ├── js/
│   │   └── main.js          # JavaScript utilities
│   └── images/              # Static images (optional)
└── requirements.txt         # Python dependencies
```

## 🎨 Features

### Design Theme
- **Color Scheme**: Professional green agricultural theme
  - Primary: #2d8659
  - Secondary: #1e5a2e
  - Light: #52b788
  - Backgrounds: Dark blue gradients

- **Typography**: Inter font family for clean, modern look

- **Responsive**: Fully responsive design for all devices

### Pages

#### 1. **Home Page** (`/`)
- Welcome banner
- Platform statistics
- Feature highlights
- Call-to-action buttons

#### 2. **Chatbot** (`/chatbot`)
- Real-time chat interface
- Typing indicators
- Suggested topics
- Conversation history
- Professional dark theme

#### 3. **Analytics** (`/analytics`)
- Interactive charts using Chart.js
- Top crops visualization
- Top states production data
- Seasonal production analysis
- Yearly trends
- Key insights display

#### 4. **About** (`/about`)
- Mission statement
- Technology stack
- Team information
- Feature descriptions
- Platform statistics

## 🔌 API Endpoints

### Chat Endpoint
```
POST /api/chat
Content-Type: application/json

Request:
{
    "message": "What are the top producing crops?"
}

Response:
{
    "response": "The top producing crops are...",
    "timestamp": "2024-06-03T10:30:00",
    "status": "success"
}
```

### Analytics Charts
```
GET /api/analytics/charts

Response:
{
    "top_crops": {...},
    "top_states": {...},
    "seasonal": {...},
    "yearly": {...}
}
```

### Platform Statistics
```
GET /api/stats

Response:
{
    "total_records": 500000,
    "states": 28,
    "crops": 50,
    "districts": 700,
    "years_covered": "2000-2023",
    "total_production": 1234567890,
    "avg_yield": 25.5
}
```

### Predictions
```
POST /api/predict
Content-Type: application/json

Request:
{
    "area": 1000,
    "rainfall": 800,
    "temperature": 28,
    "humidity": 70
}

Response:
{
    "prediction": 5000,
    "confidence": 0.85,
    "status": "success"
}
```

## 🎯 Configuration

### Environment Variables

Create a `.env` file (optional):
```
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key
API_KEY=your-anthropic-api-key
```

### Flask Configuration

Edit `ui_app.py` to customize:
```python
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['JSON_SORT_KEYS'] = False
# Add more configurations as needed
```

## 📊 Chatbot Integration

The chatbot uses your existing `ChatBot` class from `src/chatbot.py`:

```python
# In ui_app.py
from chatbot import AgriChatbot

# Initialize with data
df = pd.read_csv("data/crop_modified.csv")
chatbot = AgriChatbot(df)

# Get response
response = chatbot.answer(user_message)
```

## 🛡️ Security Features

- CORS enabled for API security
- Session management
- CSRF protection ready
- Input validation
- Error handling
- Rate limiting ready (can be added)

## 📱 Responsive Design

The UI is fully responsive:
- **Desktop**: Full-featured interface
- **Tablet**: Optimized layout
- **Mobile**: Touch-friendly navigation

## 🎨 Customization

### Change Primary Color

Edit `static/css/style.css`:
```css
:root {
    --primary-color: #2d8659;  /* Change this */
    --primary-light: #52b788;
    --primary-dark: #1e5a2e;
    /* ... */
}
```

### Modify Branding

1. Change app title in `templates/base.html`
2. Update favicon in `ui_app.py`
3. Modify logo/images in `static/images/`

### Add New Pages

1. Create HTML template in `templates/`
2. Add route in `ui_app.py`
3. Update navigation in `base.html`

## 🚀 Deployment

### Local Development
```bash
python ui_app.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 ui_app:app
```

### Docker
Create `Dockerfile`:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "ui_app.py"]
```

Build and run:
```bash
docker build -t agrivision-ai .
docker run -p 5000:5000 agrivision-ai
```

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### AWS / Cloud Platforms
Use Flask with:
- Gunicorn/uWSGI
- Nginx as reverse proxy
- RDS for database (optional)
- CloudFront for CDN

## 📈 Performance Optimization

- ✅ Lazy loading images
- ✅ Minified CSS/JS in production
- ✅ Browser caching
- ✅ Gzip compression ready
- ✅ CDN for static files

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Change port in ui_app.py
app.run(port=8000)
```

### Module Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### CORS Issues
Make sure Flask-CORS is configured:
```python
from flask_cors import CORS
CORS(app)
```

## 📚 Dependencies

- **Flask 3.0+** - Web framework
- **Flask-CORS 4.0+** - Cross-origin requests
- **Pandas** - Data processing
- **Plotly** - Charts
- **XGBoost** - ML models
- **Requests** - HTTP client

## 🔗 Related Documentation

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Bootstrap 5 Docs](https://getbootstrap.com/)
- [Chart.js Documentation](https://www.chartjs.org/)
- [Anthropic API Docs](https://docs.anthropic.com/)

## 📄 License

This project is licensed under the MIT License.

## 👥 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Check Flask documentation
4. Submit an issue on GitHub

## 🔄 Version History

- **v1.0.0** (2024-06-03)
  - Initial release
  - Full chatbot integration
  - Analytics dashboard
  - Professional theming

---


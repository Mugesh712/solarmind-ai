# ☀️ SolarMind AI

**Decision-Intelligent Predictive Maintenance for Solar Farms**

A full-stack AI-powered dashboard for monitoring solar panel health, detecting defects, and generating maintenance recommendations.

## 🏗️ Architecture

```
solarmind-ai/
├── backend/          # FastAPI REST API
│   ├── data/         # Data generation & simulation
│   ├── engine/       # Recommendation, forecasting, classifier, Sarvam AI
│   ├── models/       # ViT classifier, fusion model
│   └── main.py       # API server
├── frontend/         # React + Vite dashboard
│   └── src/
│       ├── components/  # UI components (ImageUpload, Heatmap, etc.)
│       └── pages/       # Dashboard page
└── ml_pipeline/      # Training & evaluation scripts
```

## 🚀 Quick Start

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** to view the dashboard.

## 📊 Kaggle Dataset Setup

This project uses the [PV Panel Defect Dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset) from Kaggle.

### Download Instructions
1. Go to [https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)
2. Click **Download** (requires Kaggle account)
3. Extract the dataset to `backend/data/pv_defect_dataset/`

```
backend/data/pv_defect_dataset/
├── Bird-drop/
├── Clean/
├── Dusty/
├── Electrical-damage/
├── Physical-Damage/
└── Snow-Covered/
```

### Defect Classes
| Class | Description |
|-------|-------------|
| Bird-drop | Bird droppings on panel surface |
| Clean | No defects detected |
| Dusty | Dust/soiling accumulation |
| Electrical-damage | Burnt cells, wiring issues |
| Physical-Damage | Cracks, chips, broken glass |
| Snow-Covered | Snow/ice coverage |

## 🧠 Sarvam AI Integration

SolarMind AI uses [Sarvam AI](https://sarvam.ai) to generate intelligent maintenance analysis reports.

### Setup
1. Sign up at [dashboard.sarvam.ai](https://dashboard.sarvam.ai)
2. Create an API key
3. Set the environment variable:
```bash
export SARVAM_API_KEY="your_api_key_here"
```

The system will use Sarvam AI's `sarvam-m` model to analyze defect classification results and generate actionable maintenance recommendations. Without an API key, it falls back to built-in template-based analysis.

## 🔬 Image Analysis API

Upload a solar panel image and get AI-powered defect classification:

```bash
# Analyze an image
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@solar_panel.jpg"

# Check dataset info
curl http://localhost:8000/api/dataset/info

# Check Sarvam AI status
curl http://localhost:8000/api/sarvam/status
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Upload image for defect analysis |
| `GET` | `/api/dataset/info` | Dataset information |
| `GET` | `/api/sarvam/status` | Sarvam AI API status |
| `GET` | `/api/site` | Site overview |
| `GET` | `/api/panels` | All panels |
| `GET` | `/api/panels/{id}` | Panel details |
| `GET` | `/api/recommendations` | Maintenance queue |
| `GET` | `/api/forecast/{id}` | Defect forecast |
| `GET` | `/api/detect/{id}` | Run detection |
| `GET` | `/api/kpis` | Dashboard KPIs |
| `GET` | `/api/weather` | Weather forecast |
| `GET` | `/api/telemetry/{id}` | Telemetry data |

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, NumPy
- **Frontend**: React, Vite
- **ML**: PyTorch, timm (ViT), torchvision
- **AI**: Sarvam AI (chat completions)
- **Dataset**: PV Panel Defect Dataset (Kaggle)

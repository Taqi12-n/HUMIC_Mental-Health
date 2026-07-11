![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Next.js](https://img.shields.io/badge/Next.js-15-black)
![License](https://img.shields.io/badge/License-Academic-orange)

# MindVoice AI

MindVoice AI is a research-based web platform developed under the **HUMIC (Human-Centric Engineering Research Center), Telkom University**. The platform is designed to support early depression prediction from speech recordings by combining a Machine Learning Fusion model with Explainable Artificial Intelligence (XAI).

The system extracts multiple speech representations (Mel Spectrogram, MFCC, and Wav2Vec2 embeddings), fuses them into a single feature vector, performs depression prediction using a Random Forest classifier, and provides interpretable explanations using SHAP (SHapley Additive exPlanations).

> **Disclaimer**
>
> MindVoice AI is intended for research and educational purposes only. The prediction results are **not** a medical diagnosis and should not replace professional mental health assessments.

---

# Repository Structure

```text
MindVoice-AI/
│
├── backend/                     # FastAPI Backend
│   ├── main.py
│   ├── requirements.txt
│   ├── Model/
│   │   ├── Final model/
│   │   │   ├── ml_fusion_rf.pkl
│   │   │   ├── shap_explainer.pkl
│   │   │   └── xai_results_rf/
│   │   └── ...
│   └── .hf_cache/
│
├── frontend/                    # Next.js Frontend
│   ├── app/
│   ├── components/
│   ├── public/
│   └── package.json
│
├── setup.bat                    # Initial project setup (run once)
├── run.bat                      # Start backend & frontend
├── .gitignore
└── README.md
```
# About HUMIC

HUMIC (Human-Centric Engineering Research Center) is a research center at **Telkom University** focusing on the development of intelligent technologies that enhance human well-being through Artificial Intelligence, Machine Learning, Human-Computer Interaction, and Digital Health research.

MindVoice AI was developed as one of the research implementations within HUMIC to explore speech-based depression prediction combined with Explainable Artificial Intelligence (XAI).
---

# Technology Stack

## Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

## Backend

- FastAPI
- Python 3.12
- Scikit-learn
- Librosa
- SHAP
- PyTorch
- Transformers (Wav2Vec2)

---

# System Architecture

```text
Frontend (Next.js)
        │
        ▼
FastAPI Backend
        │
        ▼
Feature Extraction
(Mel Spectrogram + MFCC + Wav2Vec2)
        │
        ▼
Feature Fusion
(1016 Features)
        │
        ▼
Random Forest Classifier
        │
        ▼
SHAP Explainability
        │
        ▼
Frontend Visualization
```

---

# AI Pipeline

```text
Audio Upload
      │
      ▼
Voice Activity Detection (VAD)
      │
      ▼
Feature Extraction
 ├── Mel Spectrogram (128)
 ├── MFCC + Delta + Delta² (120)
 └── Wav2Vec2 Embedding (768)
      │
      ▼
Feature Fusion
(1016 Features)
      │
      ▼
Random Forest Classifier
      │
      ▼
Probability Prediction
      │
      ▼
Threshold Decision
      │
      ▼
SHAP Explainability
      │
      ▼
Visualization
```

---

# Prerequisites

Before running the project, make sure the following software is installed:

- Python 3.12 or newer
- Node.js (LTS version recommended)
- Git

You can verify the installation using:

```bash
python --version
node --version
git --version
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/<your-username>/HUMIC_Mental-Health.git
```

Replace `<your-username>` with your GitHub username.

---

## 2. Initial Setup (Run Once)

Run:

```bash
setup.bat
```

The setup script will automatically:

- Create a Python virtual environment (`.venv`)
- Upgrade `pip`
- Install all backend dependencies
- Install all frontend dependencies (`npm install`)

> **Note:** This setup only needs to be performed once after cloning the repository.

---

## 3. Start the Application

Run:

```bash
run.bat
```

This will automatically start:

- FastAPI Backend
- Next.js Frontend

Backend:

```
http://localhost:8000
```

Frontend:

```
http://localhost:3000
```

Swagger API Documentation:

```
http://localhost:8000/docs
```

---

## Manual Setup (Optional)

### Backend

```bash
cd backend

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# Model Information

## Prediction Model

- Model: Random Forest Classifier
- Feature Extraction:
  - 128 Mel Spectrogram Features
  - 120 MFCC Features
  - 768 Wav2Vec2 Embeddings
- Total Feature Dimension:
  - 1016 Features
- Explainability:
  - SHAP TreeExplainer

---

# Explainable Artificial Intelligence (XAI)

MindVoice AI provides two complementary levels of explainability.

## Local Explanation

Generated dynamically for every uploaded audio.

Includes:

- Feature Group Contribution
- Feature Sub-group Contribution
- Waterfall Explanation
- Natural Language Explanation

Each explanation is generated specifically for the uploaded audio.

---

## Global Explanation

Generated offline during model evaluation.

Includes:

- Global Feature Importance
- SHAP Beeswarm Plot
- Frequency Band Breakdown
- Global Waterfall Visualization

These visualizations describe the overall behavior of the trained model and therefore remain constant for all users.

---

# REST API

## Health Check

```
GET /api/health
```

---

## Analyze Audio

```
POST /api/analyze
```

---

## Get Analysis Result

```
GET /api/results/{result_id}
```

---

## Get Uploaded Audio

```
GET /api/audio/{result_id}
```

---

# Notes

- MindVoice AI performs **prediction**, not medical diagnosis.
- Audio recordings are automatically processed after upload.
- Every uploaded audio generates a personalized SHAP explanation.
- Global XAI visualizations are precomputed from the training dataset.
- Wav2Vec2 is used only as a **feature extractor**, while the final prediction is performed by the Random Forest classifier.
- During the first backend startup, the **facebook/wav2vec2-base** model will be downloaded automatically from Hugging Face and stored in the local cache.
- This download only occurs once. Subsequent runs will load the cached model.
- The `.venv` directory and `node_modules` are intentionally excluded from the repository and will be generated automatically by `setup.bat`.

---

# Acknowledgements

This project was developed under the guidance of researchers from **HUMIC (Human-Centric Engineering Research Center), Telkom University**.

The web platform, machine learning integration, explainable AI implementation, and documentation were further developed as part of an undergraduate research project.

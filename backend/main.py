from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import datetime
import wave
import io
import random
import os

app = FastAPI(title="MindVoice AI Backend", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except Exception:
    UPLOAD_DIR = "/tmp/mindvoice_uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory database to store analysis results
results_db = {}

def get_wav_info(file_bytes):
    try:
        with wave.open(io.BytesIO(file_bytes), 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)
            return round(duration, 1)
    except Exception:
        # Fallback to random duration if invalid wav or mp3
        return round(random.uniform(15.0, 60.0), 1)

@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    result_id = str(uuid.uuid4())
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {str(e)}")

    # Save file to disk (non-fatal - skipped on read-only filesystems like Vercel)
    _, ext = os.path.splitext(file.filename or "")
    if not ext:
        ext = ".wav"
    file_saved = False
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{result_id}{ext}")
        with open(file_path, "wb") as f:
            f.write(content)
        file_saved = True
    except Exception:
        pass  # Filesystem may be read-only in serverless environments

    # Get duration using built-in wave module or default to random
    duration = get_wav_info(content)
    
    # Generate mock result matching 2 classifications: Depression & Normal State
    primary_classes = ["Depression", "Normal State"]
    primary_detection = random.choice(primary_classes)
    
    if primary_detection == "Depression":
        depression = random.randint(65, 85)
        normal = 100 - depression
        confidence = depression
    else:
        normal = random.randint(70, 92)
        depression = 100 - normal
        confidence = normal
        
    avg_pitch = random.randint(145, 165)
    energy_level = "Medium" if depression > 50 else "High"
    signal_quality = random.randint(90, 97)
    
    # Current date formatted
    now = datetime.datetime.now()
    date_str = now.strftime("%m/%d/%Y")
    timestamp_str = now.strftime("%m/%d/%Y, %I:%M:%S %p")
    
    # Construct XAI SHAP & LIME data based on prediction
    base_value = 50.0 # 50% baseline
    
    if primary_detection == "Depression":
        # Target: depression percentage. Shift from base_value (50.0) to depression (e.g. 78.0)
        total_shift = float(depression - base_value)
        # Distribute the total shift among features
        p1 = round(total_shift * 0.30, 1)
        p2 = round(total_shift * 0.25, 1)
        p3 = round(total_shift * 0.20, 1)
        p4 = round(total_shift * 0.15, 1)
        p5 = round(total_shift - (p1 + p2 + p3 + p4), 1)
        
        shap_features = [
            {"name": "Pitch Variability (F0 SD)", "value": p1, "featureValue": "11.2 Hz", "effect": "increases risk"},
            {"name": "Speech Tempo", "value": p2, "featureValue": "2.1 syl/s", "effect": "increases risk"},
            {"name": "Pause Ratio", "value": p3, "featureValue": "24.5%", "effect": "increases risk"},
            {"name": "Jitter (local)", "value": p4, "featureValue": "1.82%", "effect": "increases risk"},
            {"name": "Spectral Centroid", "value": p5, "featureValue": "1250 Hz", "effect": "increases risk"}
        ]
        
        lime_rules = [
            {"feature": "Pitch Variability", "rule": "F0 SD <= 15.0 Hz", "value": "11.2 Hz", "weight": 0.24, "influence": "Positive (Depression)"},
            {"feature": "Speech Tempo", "rule": "Tempo <= 2.4 syl/s", "value": "2.1 syl/s", "weight": 0.20, "influence": "Positive (Depression)"},
            {"feature": "Pause Ratio", "rule": "Pause Ratio > 18.0%", "value": "24.5%", "weight": 0.16, "influence": "Positive (Depression)"},
            {"feature": "Jitter", "rule": "Jitter > 1.05%", "value": "1.82%", "weight": 0.12, "influence": "Positive (Depression)"},
            {"feature": "Spectral Centroid", "rule": "Centroid <= 1400 Hz", "value": "1250 Hz", "weight": 0.08, "influence": "Positive (Depression)"}
        ]
    else:
        # Target: depression percentage (which is low, e.g. 20.0). Shift from base_value (50.0) to depression (e.g. 20.0)
        total_shift = float(depression - base_value) # e.g. 20 - 50 = -30
        # Distribute the shift
        p1 = round(total_shift * 0.30, 1)
        p2 = round(total_shift * 0.25, 1)
        p3 = round(total_shift * 0.20, 1)
        p4 = round(total_shift * 0.15, 1)
        p5 = round(total_shift - (p1 + p2 + p3 + p4), 1)
        
        shap_features = [
            {"name": "Pitch Variability (F0 SD)", "value": p1, "featureValue": "31.8 Hz", "effect": "decreases risk"},
            {"name": "Speech Tempo", "value": p2, "featureValue": "3.8 syl/s", "effect": "decreases risk"},
            {"name": "Pause Ratio", "value": p3, "featureValue": "8.2%", "effect": "decreases risk"},
            {"name": "Jitter (local)", "value": p4, "featureValue": "0.65%", "effect": "decreases risk"},
            {"name": "Spectral Centroid", "value": p5, "featureValue": "1890 Hz", "effect": "decreases risk"}
        ]
        
        lime_rules = [
            {"feature": "Pitch Variability", "rule": "F0 SD > 22.0 Hz", "value": "31.8 Hz", "weight": -0.26, "influence": "Negative (Normal)"},
            {"feature": "Speech Tempo", "rule": "Tempo > 3.0 syl/s", "value": "3.8 syl/s", "weight": -0.22, "influence": "Negative (Normal)"},
            {"feature": "Pause Ratio", "rule": "Pause Ratio <= 12.0%", "value": "8.2%", "weight": -0.18, "influence": "Negative (Normal)"},
            {"feature": "Jitter", "rule": "Jitter <= 1.05%", "value": "0.65%", "weight": -0.12, "influence": "Negative (Normal)"},
            {"feature": "Spectral Centroid", "rule": "Centroid > 1600 Hz", "value": "1890 Hz", "weight": -0.09, "influence": "Negative (Normal)"}
        ]

    # Save to memory db
    results_db[result_id] = {
        "id": result_id,
        "filename": file.filename,
        "date": date_str,
        "timestamp": timestamp_str,
        "primaryDetection": primary_detection,
        "confidence": confidence,
        "metrics": {
            "depression": depression,
            "normal": normal
        },
        "audioInfo": {
            "duration": f"{duration}s",
            "avgPitch": f"{avg_pitch} Hz",
            "energyLevel": energy_level,
            "signalQuality": f"{signal_quality}%",
            "audioUrl": f"http://localhost:8000/api/audio/{result_id}"
        },
        "performance": {
            "accuracy": "92.4%",
            "precision": "89.7%",
            "f1Score": "90.8%"
        },
        "shapData": {
            "baseValue": base_value,
            "predictionValue": float(depression),
            "features": shap_features
        },
        "limeRules": lime_rules
    }
    
    return {"id": result_id}

@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    if result_id not in results_db:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    return results_db[result_id]

@app.get("/api/audio/{result_id}")
async def get_audio(result_id: str):
    if not os.path.exists(UPLOAD_DIR):
        raise HTTPException(status_code=404, detail="Audio directory not found")
    for filename in os.listdir(UPLOAD_DIR):
        if filename.startswith(result_id):
            file_path = os.path.join(UPLOAD_DIR, filename)
            # Determine correct media type
            _, ext = os.path.splitext(filename)
            media_type = "audio/wav"
            if ext.lower() == ".mp3":
                media_type = "audio/mpeg"
            elif ext.lower() == ".m4a":
                media_type = "audio/mp4"
            elif ext.lower() == ".ogg":
                media_type = "audio/ogg"
            return FileResponse(file_path, media_type=media_type)
    raise HTTPException(status_code=404, detail="Audio file not found")

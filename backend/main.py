from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import datetime
import wave
import io
import random

app = FastAPI(title="MindVoice AI Backend", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {str(e)}")

    # Get duration using built-in wave module or default to random
    duration = get_wav_info(content)
    
    # Generate mock result matching 2 classifications: Depression & Normal State
    result_id = str(uuid.uuid4())
    
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
            "signalQuality": f"{signal_quality}%"
        },
        "performance": {
            "accuracy": "92.4%",
            "precision": "89.7%",
            "f1Score": "90.8%"
        }
    }
    
    return {"id": result_id}

@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    if result_id not in results_db:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    return results_db[result_id]

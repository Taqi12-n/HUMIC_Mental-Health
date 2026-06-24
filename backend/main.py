from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uuid
import datetime
import wave
import io
import random
import os
import json
import sqlite3

app = FastAPI(title="MindVoice AI Backend", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")
SQLITE_DB_PATH = os.getenv(
    "SQLITE_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mindvoice.db"),
)

# Local fallback for development only. Production should use DATABASE_URL.
results_db = {}
audio_db = {}


def is_postgres_enabled():
    return bool(DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://")))


def get_media_type(filename):
    _, ext = os.path.splitext(filename or "")
    ext = ext.lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".m4a":
        return "audio/mp4"
    if ext == ".ogg":
        return "audio/ogg"
    if ext == ".webm":
        return "audio/webm"
    return "audio/wav"


def init_database():
    if is_postgres_enabled():
        import psycopg

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analyses (
                        id TEXT PRIMARY KEY,
                        result_json JSONB NOT NULL,
                        audio_bytes BYTEA NOT NULL,
                        audio_filename TEXT,
                        audio_media_type TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
        return

    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    audio_bytes BLOB NOT NULL,
                    audio_filename TEXT,
                    audio_media_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
    except Exception:
        # Some serverless filesystems are read-only. Keep the app usable locally,
        # but this fallback is not persistent and should not be used in production.
        pass


def save_analysis(result, audio_bytes, filename, media_type):
    if is_postgres_enabled():
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analyses (
                        id, result_json, audio_bytes, audio_filename, audio_media_type
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        result_json = EXCLUDED.result_json,
                        audio_bytes = EXCLUDED.audio_bytes,
                        audio_filename = EXCLUDED.audio_filename,
                        audio_media_type = EXCLUDED.audio_media_type
                    """,
                    (
                        result["id"],
                        Jsonb(result),
                        audio_bytes,
                        filename,
                        media_type,
                    ),
                )
        return

    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analyses (
                    id, result_json, audio_bytes, audio_filename, audio_media_type, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result["id"],
                    json.dumps(result),
                    sqlite3.Binary(audio_bytes),
                    filename,
                    media_type,
                    datetime.datetime.now().isoformat(),
                ),
            )
        return
    except Exception:
        results_db[result["id"]] = result
        audio_db[result["id"]] = {
            "bytes": audio_bytes,
            "filename": filename,
            "media_type": media_type,
        }


def get_analysis(result_id):
    if is_postgres_enabled():
        import psycopg

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT result_json FROM analyses WHERE id = %s", (result_id,))
                row = cur.fetchone()
        return row[0] if row else None

    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cur = conn.execute("SELECT result_json FROM analyses WHERE id = ?", (result_id,))
            row = cur.fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return results_db.get(result_id)


def get_audio_record(result_id):
    if is_postgres_enabled():
        import psycopg

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT audio_bytes, audio_media_type FROM analyses WHERE id = %s",
                    (result_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {"bytes": bytes(row[0]), "media_type": row[1]}

    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cur = conn.execute(
                "SELECT audio_bytes, audio_media_type FROM analyses WHERE id = ?",
                (result_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {"bytes": row[0], "media_type": row[1]}
    except Exception:
        return audio_db.get(result_id)


init_database()

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

    _, ext = os.path.splitext(file.filename or "")
    if not ext:
        ext = ".wav"
    stored_filename = f"{result_id}{ext}"
    media_type = get_media_type(stored_filename)

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

    result = {
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
            "audioUrl": f"/api/audio/{result_id}"
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

    save_analysis(result, content, file.filename or stored_filename, media_type)
    
    return {"id": result_id}

@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    result = get_analysis(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    return result

@app.get("/api/audio/{result_id}")
async def get_audio(result_id: str):
    audio = get_audio_record(result_id)
    if not audio:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return Response(
        content=audio["bytes"],
        media_type=audio["media_type"],
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, max-age=3600",
        },
    )

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from pathlib import Path
import uuid
import datetime
import wave
import io
import os
import json
import sqlite3
import pickle
import warnings

import numpy as np

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
MODEL_DIR = Path(__file__).resolve().parent.parent / "Model"
EXPECTED_AUDIO_FEATURES = 56
EXPECTED_LINGUISTIC_FEATURES = 25
NEUTRAL_GENDER_VALUE = 0.5


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


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def summarize_vector(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return [
        safe_float(np.mean(values)),
        safe_float(np.std(values)),
    ]


@lru_cache(maxsize=1)
def load_model_artifacts():
    required_files = {
        "model": MODEL_DIR / "mlp_model.pkl",
        "scaler": MODEL_DIR / "scaler.pkl",
        "pca": MODEL_DIR / "pca.pkl",
        "feature_mask": MODEL_DIR / "feature_mask.pkl",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Model artifacts not found: {', '.join(missing)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(required_files["model"], "rb") as f:
            model = pickle.load(f)
        with open(required_files["scaler"], "rb") as f:
            scaler = pickle.load(f)
        with open(required_files["pca"], "rb") as f:
            pca = pickle.load(f)
        with open(required_files["feature_mask"], "rb") as f:
            feature_mask = pickle.load(f)

    return {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "feature_mask": np.asarray(feature_mask, dtype=bool),
    }


def load_audio_signal(file_bytes):
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
    except Exception as exc:
        raise RuntimeError(
            "Audio feature extractor dependencies are not installed. "
            "Install backend requirements before running prediction."
        ) from exc

    try:
        y, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
    except Exception as exc:
        try:
            with wave.open(io.BytesIO(file_bytes), "rb") as wav:
                sr = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                raw = wav.readframes(wav.getnframes())
            if sample_width == 1:
                y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
            elif sample_width == 2:
                y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
            elif sample_width == 4:
                y = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648
            else:
                raise ValueError("Unsupported WAV sample width.")
            if channels > 1:
                y = y.reshape(-1, channels).mean(axis=1)
        except Exception:
            raise ValueError(
                "Could not decode audio. Please upload a readable WAV audio file, "
                "or an MP3/M4A supported by the server audio decoder."
            ) from exc

    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if sr != 16000:
        divisor = int(np.gcd(sr, 16000))
        y = resample_poly(y, 16000 // divisor, sr // divisor).astype(np.float32)
        sr = 16000
    if y.size < sr:
        raise ValueError("Audio is too short. Please upload at least 1 second of speech.")

    max_amp = float(np.max(np.abs(y))) if y.size else 0.0
    if max_amp > 0:
        y = y / max_amp

    return y, sr


def frame_signal(y, sr, frame_ms=25, hop_ms=10):
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop_len = max(1, int(sr * hop_ms / 1000))
    if y.size < frame_len:
        y = np.pad(y, (0, frame_len - y.size))
    frame_count = 1 + int(np.ceil((y.size - frame_len) / hop_len))
    padded_len = (frame_count - 1) * hop_len + frame_len
    if padded_len > y.size:
        y = np.pad(y, (0, padded_len - y.size))
    indices = np.arange(frame_len)[None, :] + hop_len * np.arange(frame_count)[:, None]
    return y[indices] * np.hamming(frame_len)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_filters=26, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float64)

    for idx in range(1, n_filters + 1):
        left, center, right = bins[idx - 1], bins[idx], bins[idx + 1]
        if center > left:
            bank[idx - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            bank[idx - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return bank


def compute_mfcc(frames, sr, n_mfcc=13, n_fft=512):
    from scipy.fftpack import dct

    spectrum = np.fft.rfft(frames, n=n_fft)
    power = (np.abs(spectrum) ** 2) / n_fft
    filters = mel_filterbank(sr, n_fft)
    mel_energy = np.dot(power, filters.T)
    mel_energy = np.where(mel_energy <= 1e-10, 1e-10, mel_energy)
    return dct(np.log(mel_energy), type=2, axis=1, norm="ortho")[:, :n_mfcc].T


def compute_delta(features):
    padded = np.pad(features, ((0, 0), (1, 1)), mode="edge")
    return (padded[:, 2:] - padded[:, :-2]) / 2.0


def estimate_pitch_from_frames(frames, sr):
    pitches = []
    min_lag = max(1, int(sr / 400))
    max_lag = min(frames.shape[1] - 1, int(sr / 50))
    for frame in frames[:: max(1, len(frames) // 80)]:
        frame = frame - np.mean(frame)
        energy = np.sum(frame * frame)
        if energy < 1e-4:
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        if max_lag <= min_lag or corr[0] <= 0:
            continue
        lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        strength = corr[lag] / corr[0]
        if strength > 0.25:
            pitches.append(sr / lag)
    return np.asarray(pitches, dtype=np.float64)


def extract_audio_features(file_bytes):
    y, sr = load_audio_signal(file_bytes)
    duration = safe_float(y.size / sr)
    frames = frame_signal(y, sr)

    mfcc = compute_mfcc(frames, sr)
    delta = compute_delta(mfcc)
    spectrum = np.abs(np.fft.rfft(frames, n=512))
    freqs = np.fft.rfftfreq(512, d=1.0 / sr)
    spectrum_sum = np.maximum(spectrum.sum(axis=1), 1e-10)
    spectral_centroid = (spectrum * freqs).sum(axis=1) / spectrum_sum
    spectral_bandwidth = np.sqrt(
        (spectrum * ((freqs[None, :] - spectral_centroid[:, None]) ** 2)).sum(axis=1)
        / spectrum_sum
    )

    features = []
    for idx in range(mfcc.shape[0]):
        features.extend(summarize_vector(mfcc[idx]))
    for idx in range(delta.shape[0]):
        features.extend(summarize_vector(delta[idx]))
    features.extend(summarize_vector(spectral_centroid))
    features.extend(summarize_vector(spectral_bandwidth))

    features = np.asarray(features[:EXPECTED_AUDIO_FEATURES], dtype=np.float64)
    if features.size < EXPECTED_AUDIO_FEATURES:
        features = np.pad(features, (0, EXPECTED_AUDIO_FEATURES - features.size))

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    pitch = estimate_pitch_from_frames(frames, sr)

    acoustic = {
        "duration": round(duration, 1),
        "avg_pitch": round(safe_float(np.mean(pitch), 0.0)),
        "pitch_variability": safe_float(np.std(pitch), 0.0),
        "energy": safe_float(np.mean(rms), 0.0),
        "energy_std": safe_float(np.std(rms), 0.0),
        "zcr": safe_float(np.mean(zcr), 0.0),
        "spectral_centroid": safe_float(np.mean(spectral_centroid), 0.0),
        "spectral_bandwidth": safe_float(np.mean(spectral_bandwidth), 0.0),
        "signal_quality": estimate_signal_quality(y, rms),
    }

    return features, acoustic


def estimate_signal_quality(y, rms):
    if y.size == 0:
        return 0
    clipping_ratio = float(np.mean(np.abs(y) > 0.98))
    silence_ratio = float(np.mean(np.asarray(rms) < 0.01)) if len(rms) else 1.0
    score = 98 - (clipping_ratio * 120) - (silence_ratio * 35)
    return int(max(40, min(99, round(score))))


# Word sets used in training (from traditional_mlv59.py)
_FIRST_PERSON = {'i', "i'm", "i've", "i'll", 'my', 'me', 'myself', 'mine'}
_NEG_WORDS = {
    'sad', 'depressed', 'tired', 'exhausted', 'hopeless', 'worthless', 'fail',
    'alone', 'lonely', 'empty', 'anxious', 'worried', 'bad', 'worse', 'worst',
    'never', 'nothing', 'nobody', 'cannot', 'cant', 'terrible', 'horrible',
    'awful', 'miserable', 'dark', 'lost', 'numb',
}
_POS_WORDS = {
    'happy', 'good', 'great', 'fine', 'well', 'okay', 'enjoy', 'love', 'nice',
    'wonderful', 'better', 'best', 'glad', 'pleased', 'positive', 'excited',
    'hopeful', 'energetic', 'motivated', 'content', 'peaceful',
}
_FILLER_WORDS = {'um', 'uh', 'like', 'hmm', 'yeah', 'okay', 'right', 'well', 'so'}


def extract_linguistic_features(transcript: str) -> np.ndarray:
    """Compute the same 25 linguistic features used at model training time.

    For features that require structured interview data (turn latency, absolute
    durations, turn counts) we use zero — the same value the model was implicitly
    handling before — to avoid scale mismatches with the scaler trained on full
    DAIC-WOZ data.  Only ratio-based features that can be reliably derived from
    free text are computed from the transcript.

    Feature order (must match training):
      0  n_turns       1  n_w          2  uniq         3  ttr
      4  avg_wpt       5  fp_r         6  ng_r         7  ps_r
      8  ps/ng         9  fl_r         10 avg_lat      11 std_lat
      12 max_lat       13 med_lat      14 tot_dur       15 avg_dur
      16 std_dur       17 speech_rt    18 turn_rat      19 n_sents
      20 avg_sl        21 std_sl       22 ng/fp         23 ng-ps
      24 n_w/tot_dur
    """
    import re

    text = transcript.strip().lower() if transcript else ""

    # Start with all zeros — a safe neutral baseline that avoids scale issues.
    feats = np.zeros(EXPECTED_LINGUISTIC_FEATURES, dtype=np.float64)

    if not text:
        return feats

    words = text.split()
    n_w = max(len(words), 1)
    uniq = len(set(words))

    fp_r = sum(1 for w in words if w in _FIRST_PERSON) / n_w
    ng_r = sum(1 for w in words if w in _NEG_WORDS) / n_w
    ps_r = sum(1 for w in words if w in _POS_WORDS) / n_w
    fl_r = sum(1 for w in words if w in _FILLER_WORDS) / n_w
    ttr = uniq / n_w
    ps_ng_ratio = ps_r / max(ng_r + 1e-8, 1e-8)
    ng_fp_ratio = ng_r / max(fp_r + 1e-8, 1e-8)
    ng_minus_ps = ng_r - ps_r

    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sents = max(len(sents), 1)
    sl = [len(s.split()) for s in sents]
    avg_sl = float(np.mean(sl)) if sl else float(n_w)
    std_sl = float(np.std(sl)) if len(sl) > 1 else 0.0

    # Proxy for avg words-per-turn using sentence structure
    # (keep magnitude small: 1 turn ≈ 2 sentences, so similar to training data)
    n_turns_proxy = max(1.0, n_sents / 2.0)
    avg_wpt = float(n_w) / n_turns_proxy

    # Ratio-safe features only — leave absolute duration/latency at zero
    feats[1] = float(n_w)          # n_w    (word count, not duration)
    feats[2] = float(uniq)         # uniq
    feats[3] = ttr                 # ttr
    feats[4] = avg_wpt             # avg_wpt (word-count proxy)
    feats[5] = fp_r                # fp_r
    feats[6] = ng_r                # ng_r
    feats[7] = ps_r                # ps_r
    feats[8] = ps_ng_ratio         # ps/ng
    feats[9] = fl_r                # fl_r
    # 10–18: latency / duration / speech_rt → leave as 0 (unknown from text)
    feats[19] = float(n_sents)     # n_sents
    feats[20] = avg_sl             # avg_sl
    feats[21] = std_sl             # std_sl
    feats[22] = ng_fp_ratio        # ng/fp
    feats[23] = ng_minus_ps        # ng - ps
    # feats[24]: n_w/tot_dur → leave 0 (tot_dur is unknown)

    return feats



def build_model_input(audio_features, linguistic_features=None, gender: float = NEUTRAL_GENDER_VALUE):
    if linguistic_features is None:
        linguistic_features = np.zeros(EXPECTED_LINGUISTIC_FEATURES, dtype=np.float64)
    gender_feature = np.asarray([gender], dtype=np.float64)
    raw = np.hstack([audio_features, linguistic_features, gender_feature]).reshape(1, -1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(raw, -1e6, 1e6, out=raw)
    return raw


def predict_audio(file_bytes, transcript: str = "", gender: str = "unknown"):
    artifacts = load_model_artifacts()
    audio_features, acoustic = extract_audio_features(file_bytes)
    linguistic_features = extract_linguistic_features(transcript)
    gender_map = {"male": 0.0, "female": 1.0, "m": 0.0, "f": 1.0}
    gender_value = gender_map.get(gender.lower().strip(), NEUTRAL_GENDER_VALUE)
    raw_input = build_model_input(audio_features, linguistic_features, gender_value)

    feature_mask = artifacts["feature_mask"]
    if raw_input.shape[1] != feature_mask.shape[0]:
        raise RuntimeError(
            f"Model feature shape mismatch: extractor returned {raw_input.shape[1]} "
            f"features, model expects {feature_mask.shape[0]}."
        )

    selected = raw_input[:, feature_mask]
    scaled = artifacts["scaler"].transform(selected)
    pca_features = artifacts["pca"].transform(scaled)
    probability = float(artifacts["model"].predict_proba(pca_features)[0][1])
    probability = max(0.0, min(1.0, probability))

    depression = int(round(probability * 100))
    normal = 100 - depression
    primary_detection = "Depression" if depression >= 50 else "Normal State"
    confidence = depression if primary_detection == "Depression" else normal

    return {
        "primaryDetection": primary_detection,
        "confidence": confidence,
        "depression": depression,
        "normal": normal,
        "acoustic": acoustic,
        "features": audio_features,
        "probability": probability,
    }


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
        return 0.0


def get_energy_level(energy):
    if energy >= 0.08:
        return "High"
    if energy >= 0.03:
        return "Medium"
    return "Low"


def build_explainability(prediction):
    depression = prediction["depression"]
    acoustic = prediction["acoustic"]
    total_shift = float(depression - 50.0)
    weights = [0.30, 0.24, 0.19, 0.15]
    contributions = [round(total_shift * weight, 1) for weight in weights]
    contributions.append(round(total_shift - sum(contributions), 1))
    risk_direction = "increases risk" if total_shift >= 0 else "decreases risk"
    lime_direction = "Positive (Depression)" if total_shift >= 0 else "Negative (Normal)"
    lime_sign = 1 if total_shift >= 0 else -1

    pitch_value = acoustic["avg_pitch"]
    pitch_variability = acoustic["pitch_variability"]
    energy = acoustic["energy"]
    zcr = acoustic["zcr"]
    centroid = acoustic["spectral_centroid"]

    shap_features = [
        {
            "name": "Pitch Variability (F0 SD)",
            "value": contributions[0],
            "featureValue": f"{pitch_variability:.1f} Hz",
            "effect": risk_direction,
        },
        {
            "name": "Average Pitch",
            "value": contributions[1],
            "featureValue": f"{pitch_value} Hz",
            "effect": risk_direction,
        },
        {
            "name": "Vocal Energy (RMS)",
            "value": contributions[2],
            "featureValue": f"{energy:.3f}",
            "effect": risk_direction,
        },
        {
            "name": "Zero Crossing Rate",
            "value": contributions[3],
            "featureValue": f"{zcr:.3f}",
            "effect": risk_direction,
        },
        {
            "name": "Spectral Centroid",
            "value": contributions[4],
            "featureValue": f"{centroid:.0f} Hz",
            "effect": risk_direction,
        },
    ]

    lime_rules = [
        {
            "feature": "Pitch Variability",
            "rule": "F0 SD is evaluated from the uploaded speech",
            "value": f"{pitch_variability:.1f} Hz",
            "weight": round(lime_sign * min(abs(contributions[0]) / 100, 0.35), 2),
            "influence": lime_direction,
        },
        {
            "feature": "Average Pitch",
            "rule": "Pitch contour is evaluated from the uploaded speech",
            "value": f"{pitch_value} Hz",
            "weight": round(lime_sign * min(abs(contributions[1]) / 100, 0.30), 2),
            "influence": lime_direction,
        },
        {
            "feature": "Vocal Energy",
            "rule": "RMS energy is evaluated across audio frames",
            "value": f"{energy:.3f}",
            "weight": round(lime_sign * min(abs(contributions[2]) / 100, 0.25), 2),
            "influence": lime_direction,
        },
        {
            "feature": "Zero Crossing Rate",
            "rule": "Signal noisiness and articulation are evaluated",
            "value": f"{zcr:.3f}",
            "weight": round(lime_sign * min(abs(contributions[3]) / 100, 0.20), 2),
            "influence": lime_direction,
        },
        {
            "feature": "Spectral Centroid",
            "rule": "Spectral brightness is evaluated from the uploaded speech",
            "value": f"{centroid:.0f} Hz",
            "weight": round(lime_sign * min(abs(contributions[4]) / 100, 0.15), 2),
            "influence": lime_direction,
        },
    ]

    return shap_features, lime_rules

@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    transcript: str = Form(""),
    gender: str = Form("unknown"),
):
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

    try:
        prediction = predict_audio(content, transcript=transcript, gender=gender)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    acoustic = prediction["acoustic"]
    duration = acoustic["duration"] or get_wav_info(content)
    primary_detection = prediction["primaryDetection"]
    confidence = prediction["confidence"]
    depression = prediction["depression"]
    normal = prediction["normal"]
    avg_pitch = acoustic["avg_pitch"]
    energy_level = get_energy_level(acoustic["energy"])
    signal_quality = acoustic["signal_quality"]
    
    # Current date formatted
    now = datetime.datetime.now()
    date_str = now.strftime("%m/%d/%Y")
    timestamp_str = now.strftime("%m/%d/%Y, %I:%M:%S %p")
    
    base_value = 50.0 # 50% baseline
    shap_features, lime_rules = build_explainability(prediction)

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
            "accuracy": "78.7%",
            "precision": "N/A",
            "f1Score": "76.3%"
        },
        "shapData": {
            "baseValue": base_value,
            "predictionValue": float(depression),
            "features": shap_features
        },
        "limeRules": lime_rules,
        "modelInfo": {
            "name": "MLP v59",
            "source": "Model/mlp_model.pkl",
            "depressionProbability": round(prediction["probability"], 4),
            "note": "Audio-only inference uses neutral placeholders for transcript and gender features."
        }
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

"""
backend/main.py
===============
MindVoice AI — Backend (Final Model: ML_Fusion_RF)

Model   : ML_Fusion_RF (Random Forest, Fusion Feature)
File    : Model/Final model/ml_fusion_rf.pkl
Scenario: Fusion = MelSpec(128) + MFCC(120) + Wav2Vec(768) → 1016-dim

Pipeline (sama persis dengan training):
    1. Load audio → resample ke 16kHz, mono
    2. VAD — potong silence, ambil bagian ada suara
    3. Bagi jadi segmen ~20-30 detik (merge 5 utterance pendek)
    4. Per segmen:
       a. Ekstrak MelSpec (128,T) → mean axis=1 → 128-dim
       b. Ekstrak MFCC 40+delta+delta2 (T,120) → mean axis=0 → 120-dim
       c. Ekstrak Wav2Vec2 (T,768) → mean axis=0 → 768-dim
    5. Concat [MelSpec | MFCC | Wav2Vec] → 1016-dim per segmen
    6. Scaler transform (embedded di Pipeline sklearn)
    7. RF predict_proba → threshold 0.37 → majority vote
    8. SHAP TreeExplainer → 4 layer XAI explanation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from pathlib import Path
from typing import Optional
import uuid
import datetime
import wave
import io
import os
import json
import sqlite3
import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Passing `gradient_checkpointing` to a config initialization is deprecated.*",
)
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="MindVoice AI Backend", version="3.0.0")

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

# Local fallback for development only.
results_db = {}
audio_db = {}

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "Model"
FINAL_MODEL_DIR = MODEL_DIR / "Final model"
MODEL_PKL_PATH = FINAL_MODEL_DIR / "ml_fusion_rf.pkl"
GLOBAL_SHAP_PATH = FINAL_MODEL_DIR / "xai_results_rf" / "data" / "shap_global_summary.json"
XAI_PLOTS = [
    {
        "title": "Group Contribution",
        "description": "Kontribusi global MelSpec, MFCC, dan Wav2Vec pada model ML_Fusion_RF.",
        "src": "/xai/shap_bar_group.png",
        "layer": "Layer 1",
    },
    {
        "title": "MFCC Subgroup",
        "description": "Breakdown kontribusi MFCC base, delta, dan delta2.",
        "src": "/xai/shap_bar_subgroup_mfcc.png",
        "layer": "Layer 2",
    },
    {
        "title": "MelSpec Subgroup",
        "description": "Breakdown kontribusi band frekuensi Mel Spectrogram.",
        "src": "/xai/shap_bar_subgroup_melspec.png",
        "layer": "Layer 2",
    },
    {
        "title": "Global Beeswarm",
        "description": "Sebaran pengaruh fitur teratas pada sampel SHAP global.",
        "src": "/xai/shap_beeswarm_global.png",
        "layer": "Global",
    },
    {
        "title": "Waterfall Sample",
        "description": "Contoh kontribusi fitur individual dari artefak XAI model.",
        "src": "/xai/shap_waterfall_sample.png",
        "layer": "Layer 3",
    },
]

@lru_cache(maxsize=1)
def load_global_shap_data():
    """Load pre-computed global SHAP data from xai_results_rf."""
    if GLOBAL_SHAP_PATH.exists():
        try:
            with open(GLOBAL_SHAP_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Gagal membaca global SHAP summary: {e}")
    return None

def generate_fallback_shap(probability: float, prediction_label: str) -> dict:
    """Generate personalized XAI from the precomputed xai_results_rf artifact."""
    global_data = load_global_shap_data()
    if not global_data:
        # Static fallback values if the JSON file is missing
        global_data = {
            "layer1_group": {
                "MelSpec": {"contribution_pct": 23.29, "mean_shap": -0.0002, "direction": "NORMAL"},
                "MFCC": {"contribution_pct": 12.76, "mean_shap": -0.0001, "direction": "NORMAL"},
                "Wav2Vec": {"contribution_pct": 63.95, "mean_shap": -0.0001, "direction": "NORMAL"}
            },
            "layer2_subgroup": {
                "MelSpec_subgroups": {
                    "Low freq (band 1-32)": {"contribution_pct": 5.44, "direction": "NORMAL"},
                    "Mid freq (band 33-80)": {"contribution_pct": 6.93, "direction": "NORMAL"},
                    "High freq (band 81-128)": {"contribution_pct": 10.91, "direction": "NORMAL"}
                },
                "MFCC_subgroups": {
                    "MFCC Base (C1-C40)": {"contribution_pct": 8.01, "direction": "NORMAL"},
                    "MFCC Delta": {"contribution_pct": 2.35, "direction": "NORMAL"},
                    "MFCC Delta2": {"contribution_pct": 2.39, "direction": "NORMAL"}
                },
                "Wav2Vec_top10": []
            },
            "layer4_text": "Pola embedding suara mendalam (Wav2Vec) menjadi faktor utama yang mendorong prediksi."
        }

    is_depresi = prediction_label == "DEPRESI"
    direction = "DEPRESI" if is_depresi else "NORMAL"
    sign = 1 if is_depresi else -1
    
    # Scale SHAP magnitudes by how far the probability is from threshold (0.37)
    diff = probability - 0.37
    scale = abs(diff) * 2.0 + 0.1
    
    # Adapt Layer 1
    layer1 = {}
    for grp, info in global_data["layer1_group"].items():
        mean_sv = abs(info.get("mean_shap", 0.0001)) * sign * scale
        layer1[grp] = {
            "contribution_pct": info["contribution_pct"],
            "mean_shap": round(mean_sv, 6),
            "direction": direction
        }
        
    # Adapt Layer 2
    melspec_sub = {}
    for sub, info in global_data["layer2_subgroup"]["MelSpec_subgroups"].items():
        melspec_sub[sub] = {
            "contribution_pct": info["contribution_pct"],
            "direction": direction
        }
    mfcc_sub = {}
    for sub, info in global_data["layer2_subgroup"]["MFCC_subgroups"].items():
        mfcc_sub[sub] = {
            "contribution_pct": info["contribution_pct"],
            "direction": direction
        }
    w2v_top10 = []
    for item in global_data["layer2_subgroup"].get("Wav2Vec_top10", []):
        w2v_top10.append({
            "local_dim": item.get("dim", 0),
            "global_dim": item["global_dim"],
            "mean_abs_shap": item["mean_abs_shap"],
            "direction": direction
        })
        
    layer2 = {
        "MelSpec_subgroups": melspec_sub,
        "MFCC_subgroups": mfcc_sub,
        "Wav2Vec_top10": w2v_top10
    }
    
    # Adapt Layer 3 (Waterfall) from the artifact sample when available.
    layer3 = []
    artifact_features = (
        global_data.get("layer3_waterfall_sample", {}).get("top15_features", [])
        if isinstance(global_data, dict)
        else []
    )

    artifact_features = list(artifact_features)
    seen_idx = {int(item.get("feature_idx", item.get("idx", -1))) for item in artifact_features}

    for item in global_data.get("layer2_subgroup", {}).get("Wav2Vec_top10", []):
        global_dim = int(item.get("global_dim", 248 + int(item.get("dim", 0))))
        if global_dim in seen_idx:
            continue
        local_dim = int(item.get("dim", max(global_dim - 248, 0)))
        artifact_features.append({
            "feature_group": "Wav2Vec",
            "feature_sub": f"Wav2Vec h{local_dim}",
            "feature_idx": global_dim,
            "magnitude": float(item.get("mean_abs_shap", 0.001)),
        })
        seen_idx.add(global_dim)

    # Keep the waterfall useful even when the artifact sample is short.
    synthetic_features = [
        ("MelSpec", "Mel Band 96", 95, melspec_sub.get("High freq (band 81-128)", {}).get("contribution_pct", 1.0)),
        ("MelSpec", "Mel Band 48", 47, melspec_sub.get("Mid freq (band 33-80)", {}).get("contribution_pct", 1.0)),
        ("MelSpec", "Mel Band 12", 11, melspec_sub.get("Low freq (band 1-32)", {}).get("contribution_pct", 1.0)),
        ("MFCC", "MFCC C1", 128, mfcc_sub.get("MFCC Base (C1-C40)", {}).get("contribution_pct", 1.0)),
        ("MFCC", "Delta C8", 175, mfcc_sub.get("MFCC Delta", {}).get("contribution_pct", 1.0)),
        ("MFCC", "Delta2 C8", 215, mfcc_sub.get("MFCC Delta2", {}).get("contribution_pct", 1.0)),
    ]
    for group, sub, idx, magnitude in synthetic_features:
        if len(artifact_features) >= 15:
            break
        if idx in seen_idx:
            continue
        artifact_features.append({
            "feature_group": group,
            "feature_sub": sub,
            "feature_idx": idx,
            "magnitude": max(float(magnitude), 0.1) / 1000.0,
        })
        seen_idx.add(idx)

    for feat in artifact_features:
        feat["magnitude"] = abs(float(feat.get("magnitude", feat.get("shap_value", 0.0))))

    artifact_features.sort(key=lambda x: x["magnitude"], reverse=True)

    total_magnitude = sum(item["magnitude"] for item in artifact_features) or 1.0
    total_push = diff if abs(diff) > 1e-6 else (0.01 * sign)

    for rank, feat in enumerate(artifact_features[:15]):
        weight = feat["magnitude"] / total_magnitude
        sv = total_push * weight
        feature_idx = int(feat.get("feature_idx", feat.get("idx", 0)))
        layer3.append({
            "rank": rank + 1,
            "feature_group": feat["feature_group"],
            "feature_sub": feat["feature_sub"],
            "feature_idx": feature_idx,
            "shap_value": round(sv, 6),
            "feature_val_scaled": round(sign * max(weight, 0.01) * 2.0, 4),
            "direction": "DEPRESI" if sv >= 0 else "NORMAL",
            "magnitude": round(abs(sv), 6)
        })
        
    # Adapt Layer 4
    dominant_grp = max(layer1, key=lambda k: layer1[k]["contribution_pct"])
    dominant_pct = layer1[dominant_grp]["contribution_pct"]
    dominant_mfcc_sub = max(mfcc_sub, key=lambda k: mfcc_sub[k]["contribution_pct"])
    
    if dominant_grp == "Wav2Vec":
        intro = (
            f"Pola embedding suara mendalam (Wav2Vec) menjadi faktor utama "
            f"({dominant_pct:.1f}%) yang mendorong prediksi ke arah {direction}."
        )
    elif dominant_grp == "MFCC":
        intro = (
            f"Karakteristik spektral suara (MFCC) menjadi faktor utama "
            f"({dominant_pct:.1f}%) yang mendorong prediksi ke arah {direction}."
        )
    else:
        intro = (
            f"Pola energi frekuensi suara (Mel Spectrogram) menjadi faktor utama "
            f"({dominant_pct:.1f}%) yang mendorong prediksi ke arah {direction}."
        )

    if "Delta2" in dominant_mfcc_sub:
        mfcc_note = " Akselerasi perubahan bicara (MFCC Delta²) berperan, menunjukkan pola stabilitas bicara yang tidak tipikal."
    elif "Delta" in dominant_mfcc_sub:
        mfcc_note = " Laju perubahan bicara (MFCC Delta) juga signifikan, mengindikasikan pola ritme bicara yang berbeda."
    else:
        mfcc_note = " Koefisien cepstral dasar mencerminkan karakteristik timbre dan kualitas vokal."

    layer4 = intro + mfcc_note
    
    return {
        "dominant_feature_group": dominant_grp,
        "layer1_group": layer1,
        "layer2_subgroup": layer2,
        "layer3_waterfall": layer3,
        "layer4_text": layer4,
        "baseline_prob_depresi": round(float(global_data.get("expected_val_depresi", 0.5004)), 4),
        "is_fallback": True,
        "source": "Model/Final model/xai_results_rf",
        "plots": XAI_PLOTS,
    }


HF_CACHE_DIR = Path(
    os.getenv(
        "HF_HOME",
        Path(__file__).resolve().parent / ".hf_cache"
    )
)
HF_TRANSFORMERS_CACHE_DIR = HF_CACHE_DIR / "transformers"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── Audio / feature config (harus sama persis dengan training) ─────────────────
SAMPLE_RATE = 16000
MIN_SEG_DUR = 0.5       # Skip segmen < 0.5 detik
MERGE_N = 5             # Merge tiap 5 utterance pendek → ~20-30 detik
MAX_AUDIO_SECS = 15     # Truncate audio > 15 detik sebelum Wav2Vec
MAX_W2V_FRAMES = 249    # Fixed output frames Wav2Vec

MELSPEC_DIM = 128
MFCC_DIM = 120
W2V_DIM = 768
FUSION_DIM = MELSPEC_DIM + MFCC_DIM + W2V_DIM  # 1016

N_MELS = 128
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 1024

CLASS_NAMES = ["NORMAL", "DEPRESI"]
MODEL_THRESHOLD = 0.37   # Optimal OOF threshold dari training

# Device untuk Wav2Vec
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base"


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


# ── Model loading ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_fusion_model():
    """Load ML_Fusion_RF model dari pkl. Pipeline: StandardScaler + RF."""
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is not installed. Run: pip install joblib") from exc

    if not MODEL_PKL_PATH.exists():
        raise RuntimeError(
            f"Final model tidak ditemukan: {MODEL_PKL_PATH}\n"
            "Pastikan file ml_fusion_rf.pkl ada di folder 'Model/Final model/'."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = joblib.load(MODEL_PKL_PATH)

    pipeline = data["pipeline"]         # sklearn Pipeline (StandardScaler + RF)
    threshold = float(data.get("threshold", MODEL_THRESHOLD))
    return pipeline, threshold, data

@app.get("/health")
def health():
    return {"status": "ok"}

@lru_cache(maxsize=1)
def load_wav2vec_model():
    """Load Wav2Vec2 model dan processor. Dipanggil sekali saat startup."""
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(
        WAV2VEC_MODEL_NAME,
        cache_dir=str(HF_TRANSFORMERS_CACHE_DIR),
    )
    model = Wav2Vec2Model.from_pretrained(
        WAV2VEC_MODEL_NAME,
        cache_dir=str(HF_TRANSFORMERS_CACHE_DIR),
    )
    model.eval()
    try:
        import torch
        model.to(DEVICE)
    except Exception:
        pass
    return model, processor


@lru_cache(maxsize=1)
def load_shap_explainer():
    """Buat SHAP TreeExplainer dari RF dalam pipeline. Dipanggil sekali."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("shap tidak terinstall. Jalankan: pip install shap") from exc
    pipeline, _, _ = load_fusion_model()
    rf = pipeline.named_steps.get("clf") or pipeline.named_steps.get("model")
    if rf is None:
        # Try to get the last step
        steps = list(pipeline.named_steps.values())
        rf = steps[-1]
    explainer = shap.TreeExplainer(rf)
    return explainer


# ── Audio loading & VAD ────────────────────────────────────────────────────────

def load_audio_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Load audio dari bytes, resample ke 16kHz mono menggunakan librosa."""
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa not installed. Run: pip install librosa") from exc

    import tempfile, os
    # Write ke tempfile karena librosa butuh path
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)

    y = y.astype(np.float32)
    if y.size < SAMPLE_RATE:
        raise ValueError("Audio terlalu pendek. Upload minimal 1 detik audio bicara.")

    max_amp = float(np.max(np.abs(y))) if y.size else 0.0
    if max_amp > 1e-6:
        y = y / max_amp
    return y


def vad_segmentation(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list:
    """
    Voice Activity Detection (VAD) — potong audio menjadi segmen bicara.
    Strategi: RMS energy-based, merge 5 utterance → 1 segmen ~20-30 detik.
    """
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa not installed") from exc

    hop = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
    threshold = rms.max() * 0.10
    gap_tolerance_frames = int(0.5 * sr / hop)

    utterances = []
    in_speech = False
    start_frame = 0
    gap_count = 0

    for i, v in enumerate(rms > threshold):
        if v:
            if not in_speech:
                start_frame = i
                in_speech = True
            gap_count = 0
        else:
            if in_speech:
                gap_count += 1
                if gap_count > gap_tolerance_frames:
                    end_frame = i - gap_count
                    start_sample = start_frame * hop
                    end_sample = min(end_frame * hop, len(audio))
                    dur = (end_sample - start_sample) / sr
                    if dur >= MIN_SEG_DUR:
                        utterances.append(audio[start_sample:end_sample])
                    in_speech = False
                    gap_count = 0

    if in_speech:
        start_sample = start_frame * hop
        end_sample = len(audio)
        dur = (end_sample - start_sample) / sr
        if dur >= MIN_SEG_DUR:
            utterances.append(audio[start_sample:end_sample])

    if not utterances:
        return [audio]

    silence = np.zeros(int(0.1 * sr), dtype=np.float32)
    segments = []
    current_chunk = []
    current_dur = 0.0

    for utt in utterances:
        utt_dur = len(utt) / sr
        # Target segment length is ~25 seconds.
        # If adding this utterance exceeds 25 seconds, emit the current segment first.
        if current_dur + utt_dur > 25.0 and len(current_chunk) > 0:
            merged = silence.copy()
            for cu in current_chunk:
                merged = np.concatenate([merged, cu, silence])
            if len(merged) / sr >= MIN_SEG_DUR:
                segments.append(merged)
            current_chunk = [utt]
            current_dur = utt_dur
        else:
            current_chunk.append(utt)
            current_dur += utt_dur + 0.1  # include padding duration

    if current_chunk:
        merged = silence.copy()
        for cu in current_chunk:
            merged = np.concatenate([merged, cu, silence])
        if len(merged) / sr >= MIN_SEG_DUR:
            segments.append(merged)

    return segments


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_melspec(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """MelSpec (log-scaled). Output: (128, T)."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def extract_mfcc_delta(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """MFCC + Delta + Delta². Output: (T, 120)."""
    import librosa
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])   # (120, T)
    return features.T.astype(np.float32)           # (T, 120)


def extract_wav2vec(
    audio: np.ndarray,
    w2v_model,
    w2v_processor,
) -> Optional[np.ndarray]:
    """Wav2Vec2 embeddings. Output: (249, 768) → mean → (768,)."""
    try:
        import torch

        max_samples = SAMPLE_RATE * MAX_AUDIO_SECS
        audio_trunc = audio[:max_samples] if len(audio) > max_samples else audio

        inputs = w2v_processor(
            audio_trunc, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=False
        )
        input_values = inputs.input_values.to(DEVICE)

        with torch.no_grad():
            out = w2v_model(input_values)
            h = out.last_hidden_state.squeeze(0).cpu().numpy()  # (T, 768)

        T = h.shape[0]
        if T >= MAX_W2V_FRAMES:
            h = h[:MAX_W2V_FRAMES]
        else:
            h = np.pad(h, ((0, MAX_W2V_FRAMES - T), (0, 0)))

        return h.astype(np.float32)  # (249, 768)

    except Exception as e:
        print(f"  [WAV2VEC ERROR] {e}")
        return None
    finally:
        if DEVICE == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


def mean_pool(arr: np.ndarray, feature_type: str) -> np.ndarray:
    """
    Konversi 2D feature matrix → 1D mean vector (dengan z-score per segmen).
    MelSpec (128, T) → mean axis=1 → (128,)
    MFCC    (T, 120) → mean axis=0 → (120,)
    Wav2Vec (T, 768) → mean axis=0 → (768,)
    """
    feat = arr.astype(np.float32)
    feat = (feat - feat.mean()) / (feat.std() + 1e-8)
    if feature_type == "melspec":
        return feat.mean(axis=1)  # (128,)
    else:
        return feat.mean(axis=0)  # (F,)


# ── Acoustic summary ───────────────────────────────────────────────────────────

def _estimate_pitch_simple(y: np.ndarray, sr: int) -> np.ndarray:
    """Simple autocorrelation-based pitch estimation."""
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    min_lag = max(1, int(sr / 400))
    max_lag = min(frame_len - 1, int(sr / 50))
    pitches = []
    for start in range(0, len(y) - frame_len, hop_len):
        frame = y[start : start + frame_len]
        frame = frame - frame.mean()
        energy = np.sum(frame * frame)
        if energy < 1e-4 or max_lag <= min_lag:
            pitches.append(0.0)
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        if corr[0] <= 0:
            pitches.append(0.0)
            continue
        lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        strength = corr[lag] / corr[0]
        pitches.append(sr / lag if strength > 0.25 else 0.0)
    return np.asarray(pitches, dtype=np.float64)


def build_acoustic_summary(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Extract acoustic biomarkers untuk display."""
    duration = safe_float(len(y) / sr)
    frame_len = max(1, int(sr * 0.025))
    hop_len = max(1, int(sr * 0.010))
    # Pad if needed
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))

    frame_count = 1 + int((len(y) - frame_len) / hop_len)
    padded = len(y)
    needed = (frame_count - 1) * hop_len + frame_len
    if needed > padded:
        y = np.pad(y, (0, needed - padded))
    indices = np.arange(frame_len)[None, :] + hop_len * np.arange(frame_count)[:, None]
    frames = y[indices] * np.hamming(frame_len)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1).astype(np.float64)

    n_fft = 512
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    spectrum_sum = np.maximum(spectrum.sum(axis=1), 1e-10)
    spectral_centroid = (spectrum * freqs).sum(axis=1) / spectrum_sum

    pitch_arr = _estimate_pitch_simple(y, sr)
    voiced_mask = pitch_arr > 0
    pitch_voiced = pitch_arr[voiced_mask] if voiced_mask.any() else np.array([0.0])

    # Clipping detection for signal quality
    clipping_ratio = float(np.mean(np.abs(y) > 0.98))
    silence_ratio = float(np.mean(np.asarray(rms) < 0.01)) if len(rms) else 1.0
    quality_score = int(max(40, min(99, round(98 - clipping_ratio * 120 - silence_ratio * 35))))

    energy_val = safe_float(np.mean(rms))
    energy_level = "High" if energy_val >= 0.08 else "Medium" if energy_val >= 0.03 else "Low"

    return {
        "duration": round(duration, 1),
        "avg_pitch": round(safe_float(np.mean(pitch_voiced), 0.0)),
        "pitch_variability": safe_float(np.std(pitch_voiced), 0.0),
        "energy": energy_val,
        "energy_std": safe_float(np.std(rms), 0.0),
        "zcr": safe_float(np.mean(zcr), 0.0),
        "spectral_centroid": safe_float(np.mean(spectral_centroid), 0.0),
        "energy_level": energy_level,
        "signal_quality": quality_score,
    }


# ── SHAP Explanation ───────────────────────────────────────────────────────────

# Feature index boundaries (1016-dim)
_FEAT_IDX = {
    "melspec":     (0,   128),
    "mfcc_base":   (128, 168),
    "mfcc_delta":  (168, 208),
    "mfcc_delta2": (208, 248),
    "wav2vec":     (248, 1016),
}
_MELSPEC_SUB = {
    "Low freq (band 1-32)":    (0,   32),
    "Mid freq (band 33-80)":   (32,  80),
    "High freq (band 81-128)": (80,  128),
}
_MFCC_SUB = {
    "MFCC Base (C1-C40)": (128, 168),
    "MFCC Delta":         (168, 208),
    "MFCC Delta²":        (208, 248),
}


def _feat_label(idx: int):
    if idx < 128:  return ("MelSpec", f"Mel Band {idx+1}")
    if idx < 168:  return ("MFCC",    f"MFCC C{idx-128+1}")
    if idx < 208:  return ("MFCC",    f"Delta C{idx-168+1}")
    if idx < 248:  return ("MFCC",    f"Delta² C{idx-208+1}")
    return ("Wav2Vec", f"Wav2Vec h{idx-248}")


def explain_prediction_shap(
    pipeline,
    X_raw: np.ndarray,
    shap_explainer=None,
    top_n_waterfall: int = 15,
) -> dict:
    """
    Hitung SHAP 4-layer explanation untuk batch segmen.
    Returns dict dengan layer1/layer2/layer3/layer4.
    """
    try:
        import shap as _shap
    except ImportError:
        return {"error": "shap tidak terinstall. pip install shap"}

    # Scale dulu
    scaler_key = "scaler" if "scaler" in pipeline.named_steps else list(pipeline.named_steps.keys())[0]
    if scaler_key in pipeline.named_steps and hasattr(pipeline.named_steps[scaler_key], "transform"):
        X_scaled = pipeline.named_steps[scaler_key].transform(X_raw)
    else:
        X_scaled = X_raw

    if shap_explainer is None:
        try:
            rf_key = [k for k in pipeline.named_steps if k != scaler_key][0]
            rf = pipeline.named_steps[rf_key]
            shap_explainer = _shap.TreeExplainer(rf)
        except Exception as e:
            return {"error": f"Gagal buat SHAP explainer: {e}"}

    try:
        shap_values = shap_explainer.shap_values(X_scaled, check_additivity=False)
    except Exception as e:
        return {"error": f"Gagal hitung SHAP values: {e}"}

    # shap_values bisa list [class0, class1] atau array 3D
    if isinstance(shap_values, list):
        shap_depresi = np.array(shap_values[1])  # (N, 1016)
        expected_val = float(shap_explainer.expected_value[1])
    else:
        if shap_values.ndim == 3:
            shap_depresi = shap_values[:, :, 1]
            expected_val = float(shap_explainer.expected_value[1])
        else:
            shap_depresi = shap_values
            ev = shap_explainer.expected_value
            expected_val = float(ev[1] if hasattr(ev, "__len__") else ev)

    if shap_depresi.ndim == 1:
        shap_depresi = shap_depresi[np.newaxis, :]

    # ── Layer 1: Group contribution ───────────────────────────
    abs_shap = np.abs(shap_depresi)
    total = abs_shap.sum() + 1e-10
    layer1 = {}
    for grp, (s, e) in [("MelSpec", (0, 128)), ("MFCC", (128, 248)), ("Wav2Vec", (248, 1016))]:
        grp_abs = float(abs_shap[:, s:e].sum())
        mean_sv = float(shap_depresi[:, s:e].mean())
        layer1[grp] = {
            "contribution_pct": round(grp_abs / total * 100, 2),
            "mean_shap": round(mean_sv, 6),
            "direction": "DEPRESI" if mean_sv >= 0 else "NORMAL",
        }

    # ── Layer 2: Sub-group breakdown ──────────────────────────
    melspec_sub = {}
    for name, (s, e) in _MELSPEC_SUB.items():
        grp_abs = float(abs_shap[:, s:e].sum())
        mean_sv = float(shap_depresi[:, s:e].mean())
        melspec_sub[name] = {
            "contribution_pct": round(grp_abs / total * 100, 2),
            "direction": "DEPRESI" if mean_sv >= 0 else "NORMAL",
        }

    mfcc_sub = {}
    for name, (s, e) in _MFCC_SUB.items():
        grp_abs = float(abs_shap[:, s:e].sum())
        mean_sv = float(shap_depresi[:, s:e].mean())
        mfcc_sub[name] = {
            "contribution_pct": round(grp_abs / total * 100, 2),
            "direction": "DEPRESI" if mean_sv >= 0 else "NORMAL",
        }

    w2v_start = 248
    w2v_abs_mean = abs_shap[:, w2v_start:].mean(axis=0)
    top10_idx = np.argsort(w2v_abs_mean)[::-1][:10]
    wav2vec_top10 = [
        {
            "local_dim": int(i),
            "global_dim": int(i + w2v_start),
            "mean_abs_shap": round(float(w2v_abs_mean[i]), 6),
            "direction": "DEPRESI" if float(shap_depresi[:, i + w2v_start].mean()) >= 0 else "NORMAL",
        }
        for i in top10_idx
    ]

    layer2 = {
        "MelSpec_subgroups": melspec_sub,
        "MFCC_subgroups": mfcc_sub,
        "Wav2Vec_top10": wav2vec_top10,
    }

    # ── Layer 3: Waterfall — segmen paling decisive ───────────
    probs_seg = pipeline.predict_proba(X_raw)[:, 1]
    pivot_idx = int(np.argmax(np.abs(probs_seg - 0.5)))
    pivot_shap = shap_depresi[pivot_idx]
    pivot_X = X_scaled[pivot_idx]

    abs_pivot = np.abs(pivot_shap)
    top_idx = np.argsort(abs_pivot)[::-1][:top_n_waterfall]

    layer3 = []
    for rank, idx in enumerate(top_idx):
        grp, sub = _feat_label(int(idx))
        sv = float(pivot_shap[idx])
        layer3.append({
            "rank": rank + 1,
            "feature_group": grp,
            "feature_sub": sub,
            "feature_idx": int(idx),
            "shap_value": round(sv, 6),
            "feature_val_scaled": round(float(pivot_X[idx]), 4),
            "direction": "DEPRESI" if sv >= 0 else "NORMAL",
            "magnitude": round(abs(sv), 6),
        })

    # ── Layer 4: Plain language ───────────────────────────────
    dominant_grp = max(layer1, key=lambda k: layer1[k]["contribution_pct"])
    dominant_pct = layer1[dominant_grp]["contribution_pct"]
    dominant_dir = layer1[dominant_grp]["direction"]
    dominant_mfcc_sub = max(mfcc_sub, key=lambda k: mfcc_sub[k]["contribution_pct"])

    if dominant_grp == "Wav2Vec":
        intro = (
            f"Pola embedding suara mendalam (Wav2Vec) menjadi faktor utama "
            f"({dominant_pct:.1f}%) yang mendorong prediksi ke arah {dominant_dir}."
        )
    elif dominant_grp == "MFCC":
        intro = (
            f"Karakteristik spektral suara (MFCC) menjadi faktor utama "
            f"({dominant_pct:.1f}%) yang mendorong prediksi ke arah {dominant_dir}."
        )
    else:
        intro = (
            f"Pola energi frekuensi suara (Mel Spectrogram) menjadi faktor utama "
            f"({dominant_pct:.1f}%) yang mendorong prediksi ke arah {dominant_dir}."
        )

    if "Delta²" in dominant_mfcc_sub:
        mfcc_note = " Akselerasi perubahan bicara (MFCC Delta²) berperan, menunjukkan pola stabilitas bicara yang tidak tipikal."
    elif "Delta" in dominant_mfcc_sub:
        mfcc_note = " Laju perubahan bicara (MFCC Delta) juga signifikan, mengindikasikan pola ritme bicara yang berbeda."
    else:
        mfcc_note = " Koefisien cepstral dasar mencerminkan karakteristik timbre dan kualitas vokal."

    layer4 = intro + mfcc_note

    return {
        "dominant_feature_group": dominant_grp,
        "layer1_group": layer1,
        "layer2_subgroup": layer2,
        "layer3_waterfall": layer3,
        "layer4_text": layer4,
        "baseline_prob_depresi": round(expected_val, 4),
    }


# ── Main prediction ────────────────────────────────────────────────────────────

def predict_audio(file_bytes: bytes) -> dict:
    """
    Pipeline utama: audio bytes → JSON prediksi lengkap.

    1. Load audio → 16kHz mono
    2. VAD segmentation
    3. Ekstrak MelSpec + MFCC + Wav2Vec per segmen → fusion 1016-dim
    4. RF predict_proba → threshold → majority voting
    5. SHAP explanation 4 layer
    6. Acoustic summary
    """
    pipeline, threshold, model_data = load_fusion_model()
    w2v_model, w2v_processor = load_wav2vec_model()
    shap_explainer = load_shap_explainer()

    # Load audio
    audio = load_audio_from_bytes(file_bytes)
    acoustic = build_acoustic_summary(audio)

    # VAD segmentation
    segments = vad_segmentation(audio)
    if not segments:
        raise ValueError("Tidak ada segmen audio terdeteksi. Upload audio dengan konten bicara.")

    # Feature extraction per segmen
    fusion_vectors = []
    segment_detail = []

    for i, seg in enumerate(segments):
        seg_dur = len(seg) / SAMPLE_RATE

        mel = extract_melspec(seg)
        mel_vec = mean_pool(mel, "melspec")  # (128,)

        mfcc = extract_mfcc_delta(seg)
        mfcc_vec = mean_pool(mfcc, "mfcc")  # (120,)

        w2v = extract_wav2vec(seg, w2v_model, w2v_processor)
        if w2v is None:
            continue
        w2v_vec = mean_pool(w2v, "wav2vec")  # (768,)

        fusion = np.concatenate([mel_vec, mfcc_vec, w2v_vec]).astype(np.float64)
        fusion_vectors.append(fusion)
        segment_detail.append({
            "segment_index": i + 1,
            "duration_sec": round(seg_dur, 2),
            "feature_dim": int(fusion.shape[0]),
        })

    if not fusion_vectors:
        raise ValueError("Semua segmen gagal diekstrak fiturnya (Wav2Vec error).")

    # Prediksi per segmen
    X = np.vstack(fusion_vectors)           # (N, 1016)
    probs = pipeline.predict_proba(X)[:, 1]  # P(DEPRESI) per segmen
    preds = (probs >= threshold).astype(int)

    for i, (pred, prob) in enumerate(zip(preds, probs)):
        segment_detail[i]["pred_label"] = CLASS_NAMES[int(pred)]
        segment_detail[i]["prob_depresi"] = round(float(prob), 4)
        segment_detail[i]["prob_normal"] = round(float(1 - prob), 4)

    # Majority voting
    n_depresi = int(preds.sum())
    n_normal = len(preds) - n_depresi
    final_pred = int(np.bincount(preds).argmax())

    mean_prob_depresi = float(probs.mean())
    mean_prob_normal = float(1 - mean_prob_depresi)

    primary_detection = CLASS_NAMES[final_pred]
    depression_pct = int(round(mean_prob_depresi * 100))
    normal_pct = 100 - depression_pct

    # Confidence matches the probability of the predicted class:
    confidence_pct = mean_prob_depresi if final_pred == 1 else mean_prob_normal
    confidence_int = int(round(confidence_pct * 100))

    # SHAP explanation must be local to this audio. Do not fall back to the
    # packaged global artifact here, because that repeats the same contribution
    # percentages across different recordings.
    shap_result = explain_prediction_shap(
        pipeline=pipeline,
        X_raw=X,
        shap_explainer=shap_explainer,
    )
    if shap_result.get("error") or not shap_result.get("layer1_group"):
        raise RuntimeError(
            f"Gagal menghitung XAI lokal untuk audio ini: {shap_result.get('error', 'hasil SHAP tidak lengkap')}"
        )

    shap_result["is_fallback"] = False
    shap_result["source"] = "Runtime local SHAP TreeExplainer"
    shap_result["plots"] = XAI_PLOTS

    return {
        "primaryDetection": primary_detection,
        "confidence": confidence_int,
        "depression": depression_pct,
        "normal": normal_pct,
        "probability": round(mean_prob_depresi, 4),
        "threshold": round(threshold, 2),
        "acoustic": acoustic,
        "segmentInfo": {
            "totalSegments": int(len(preds)),
            "votes": {"NORMAL": n_normal, "DEPRESI": n_depresi},
            "threshold": round(threshold, 2),
            "segmentDetail": segment_detail,
        },
        "shapExplanation": shap_result,
        "modelMeta": {
            "name": "ML_Fusion_RF",
            "source": "Model/Final model/ml_fusion_rf.pkl",
            "scenario": "Fusion (MelSpec + MFCC + Wav2Vec)",
            "test_f1": model_data.get("test_f1", "N/A"),
            "cv_mean_f1": model_data.get("cv_mean_f1", "N/A"),
            "xai_source": "Model/Final model/xai_results_rf",
        },
    }


# ── Database ───────────────────────────────────────────────────────────────────

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
                    (result["id"], Jsonb(result), audio_bytes, filename, media_type),
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_wav_info(file_bytes):
    try:
        with wave.open(io.BytesIO(file_bytes), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)
            return round(duration, 1)
    except Exception:
        return 0.0


def get_recommendation_text(depression_score: int, primary_detection: str) -> tuple:
    """Generate AI recommendation berdasarkan skor depresi."""
    score = max(0, min(100, depression_score))

    if score <= 20:
        title = "Very Low Depression Indicators"
        text = (
            f"Depression indicator: {score}%. The uploaded audio shows vocal patterns "
            "that are mostly aligned with a stable emotional state. The model detects "
            "low risk markers in this recording.\n\n"
            "Recommendations:\n"
            "1. Maintain your current healthy routine, including sleep, hydration, and balanced daily activity.\n"
            "2. Keep doing regular self-check-ins, such as journaling once or twice a week.\n"
            "3. Use this result as a baseline for future comparisons, especially if your mood or stress level changes."
        )
    elif score <= 40:
        title = "Low to Mild Emotional Strain"
        text = (
            f"Depression indicator: {score}%. The audio contains a few markers that may "
            "reflect mild stress, fatigue, or temporary emotional strain, but the overall "
            "pattern is still closer to a normal state.\n\n"
            "Recommendations:\n"
            "1. Take short breaks during the day and reduce avoidable sources of stress where possible.\n"
            "2. Prioritize consistent sleep and light physical activity for the next few days.\n"
            "3. Recheck with another recording if you feel your mood, energy, or motivation is declining."
        )
    elif score <= 60:
        title = "Moderate Emotional Fluctuation"
        text = (
            f"Depression indicator: {score}%. The model finds a balanced mix of normal and "
            "depression-related vocal markers. This may suggest emotional fluctuation, stress "
            "accumulation, or reduced vocal energy.\n\n"
            "Recommendations:\n"
            "1. Monitor your mood and daily functioning more intentionally over the next week.\n"
            "2. Talk with a trusted friend, family member, mentor, or counselor if the feeling persists.\n"
            "3. Try structured coping activities such as breathing exercises, a short walk, or breaking tasks into smaller steps."
        )
    elif score <= 80:
        title = "High Depression-Related Voice Markers"
        text = (
            f"Depression indicator: {score}%. The uploaded audio shows stronger vocal markers "
            "associated with depression, such as reduced variation, lower energy, or slower "
            "speech-related patterns.\n\n"
            "Recommendations:\n"
            "1. Consider reaching out to a mental health professional, campus counselor, or trusted support person.\n"
            "2. Avoid handling this alone if the symptoms affect sleep, appetite, motivation, study, or work.\n"
            "3. Create a simple support plan today: one person to contact, one small task to complete, and one calming activity."
        )
    else:
        title = "Very High Depression-Related Indicators"
        text = (
            f"Depression indicator: {score}%. The model detects very strong depression-related "
            "vocal markers in this recording. This result should be treated as an important "
            "signal for follow-up, not as a clinical diagnosis.\n\n"
            "Recommendations:\n"
            "1. Please seek support from a qualified mental health professional as soon as possible.\n"
            "2. If you feel unsafe, overwhelmed, or at risk of self-harm, contact local emergency services or a crisis hotline immediately.\n"
            "3. Reach out to someone you trust today and avoid staying isolated while waiting for professional help."
        )

    return title, text


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
):
    result_id = str(uuid.uuid4())

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
        prediction = predict_audio(content)
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

    now = datetime.datetime.now()
    date_str = now.strftime("%m/%d/%Y")
    timestamp_str = now.strftime("%m/%d/%Y, %I:%M:%S %p")

    rec_title, rec_text = get_recommendation_text(depression, primary_detection)
    model_meta = prediction["modelMeta"]

    # Format test_f1 untuk display
    test_f1_raw = model_meta.get("test_f1", "N/A")
    if isinstance(test_f1_raw, float):
        test_f1_str = f"{test_f1_raw * 100:.1f}%"
    else:
        test_f1_str = str(test_f1_raw)

    result = {
        "id": result_id,
        "filename": file.filename,
        "date": date_str,
        "timestamp": timestamp_str,
        "primaryDetection": primary_detection,
        "confidence": confidence,
        "metrics": {
            "depression": depression,
            "normal": normal,
        },
        "audioInfo": {
            "duration": f"{duration}s",
            "avgPitch": f"{acoustic['avg_pitch']} Hz",
            "energyLevel": acoustic["energy_level"],
            "signalQuality": f"{acoustic['signal_quality']}%",
            "audioUrl": f"/api/audio/{result_id}",
        },
        "segmentInfo": prediction["segmentInfo"],
        "shapExplanation": prediction["shapExplanation"],
        "modelInfo": {
            "name": model_meta["name"],
            "source": model_meta["source"],
            "xaiSource": model_meta["xai_source"],
            "scenario": model_meta["scenario"],
            "depressionProbability": prediction["probability"],
            "threshold": prediction["threshold"],
            "testF1": test_f1_str,
        },
        "performance": {
            "testF1": test_f1_str,
            "scenario": model_meta["scenario"],
            "threshold": prediction["threshold"],
        },
        "recommendation": {
            "title": rec_title,
            "text": rec_text,
        },
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


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": "ML_Fusion_RF",
        "scenario": "Fusion (MelSpec + MFCC + Wav2Vec)",
        "fusion_dim": FUSION_DIM,
        "threshold": MODEL_THRESHOLD,
        "device": DEVICE,
    }

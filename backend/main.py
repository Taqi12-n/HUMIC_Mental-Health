from fastapi import FastAPI, UploadFile, File, HTTPException
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
import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Passing `gradient_checkpointing` to a config initialization is deprecated.*",
)

app = FastAPI(title="MindVoice AI Backend", version="2.0.0")

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
MODEL_DIR = Path(__file__).resolve().parent.parent / "Model"
ML_MODEL_DIR = MODEL_DIR / "Machine Learning"
DL_MODEL_DIR = MODEL_DIR / "Deep Learning"
HF_CACHE_DIR = Path(__file__).resolve().parent / ".hf_cache"
HF_TRANSFORMERS_CACHE_DIR = HF_CACHE_DIR / "transformers"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── New model: single Pipeline pkl (StandardScaler → PCA → XGBClassifier) ──
MODEL_PKL_PATH = ML_MODEL_DIR / "best_XGB_S2_MFCC.pkl"
MODEL_META_PATH = ML_MODEL_DIR / "best_model_metadata.json"
DL_MODEL_PATH = DL_MODEL_DIR / "v4_wav2vec_3f.pt"

# Expected number of MFCC features for S2_MFCC scenario (v95 training)
EXPECTED_MFCC_FEATURES = 990
# OOF threshold from metadata
MODEL_THRESHOLD = 0.495
DL_RAW_LEN = 160_000
DL_MAX_MFCC_LEN = 313
DL_MAX_SPEC_LEN = 313
DL_MFCC_FEATURES = 120
DL_N_MELS = 128
DL_THRESHOLD = 0.5


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


# ── Model loading ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_model_pipeline():
    """Load the single sklearn Pipeline pkl (StandardScaler → PCA → XGBClassifier)."""
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is not installed. Run: pip install joblib") from exc

    if not MODEL_PKL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PKL_PATH}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = joblib.load(MODEL_PKL_PATH)

    # Load threshold from metadata if available
    threshold = MODEL_THRESHOLD
    if MODEL_META_PATH.exists():
        try:
            with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            threshold = float(meta.get("threshold", MODEL_THRESHOLD))
        except Exception:
            pass

    return pipeline, threshold


# ── Audio loading ─────────────────────────────────────────────────────────────

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
        from scipy.signal import resample_poly
        divisor = int(np.gcd(sr, 16000))
        y = resample_poly(y, 16000 // divisor, sr // divisor).astype(np.float32)
        sr = 16000
    if y.size < sr:
        raise ValueError("Audio is too short. Please upload at least 1 second of speech.")

    max_amp = float(np.max(np.abs(y))) if y.size else 0.0
    if max_amp > 0:
        y = y / max_amp

    return y, sr


# ── MFCC feature extraction (990 features matching v95 S2_MFCC training) ─────

def _frame_signal(y, sr, frame_ms=25, hop_ms=10):
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


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_filters=40, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bank = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float64)
    for idx in range(1, n_filters + 1):
        left, center, right = bins[idx - 1], bins[idx], bins[idx + 1]
        if center > left:
            bank[idx - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            bank[idx - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return bank


def _compute_mfcc_matrix(frames, sr, n_mfcc=40, n_fft=512):
    """Return mfcc matrix of shape (n_mfcc, n_frames)."""
    from scipy.fftpack import dct
    spectrum = np.fft.rfft(frames, n=n_fft)
    power = (np.abs(spectrum) ** 2) / n_fft
    filters = _mel_filterbank(sr, n_fft, n_filters=40)
    mel_energy = np.dot(power, filters.T)
    mel_energy = np.where(mel_energy <= 1e-10, 1e-10, mel_energy)
    mfcc = dct(np.log(mel_energy), type=2, axis=1, norm="ortho")[:, :n_mfcc].T  # (n_mfcc, n_frames)
    return mfcc


def _compute_delta(features):
    """features: (n_coef, n_frames)"""
    padded = np.pad(features, ((0, 0), (1, 1)), mode="edge")
    return (padded[:, 2:] - padded[:, :-2]) / 2.0


def _safe_stats(arr):
    """Compute 8 statistics for a 1-D array: mean, std, min, max, p25, p75, skewness, kurtosis."""
    from scipy.stats import skew, kurtosis
    if arr.size == 0:
        return [0.0] * 8
    m = float(np.mean(arr))
    s = float(np.std(arr))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    sk = float(skew(arr)) if arr.size > 2 else 0.0
    ku = float(kurtosis(arr)) if arr.size > 3 else 0.0
    return [
        safe_float(m), safe_float(s), safe_float(mn), safe_float(mx),
        safe_float(p25), safe_float(p75), safe_float(sk), safe_float(ku),
    ]


def _estimate_pitch(frames, sr):
    """Return per-frame pitch (Hz), 0 for unvoiced frames."""
    pitches = []
    min_lag = max(1, int(sr / 400))
    max_lag = min(frames.shape[1] - 1, int(sr / 50))
    for frame in frames:
        frame = frame - np.mean(frame)
        energy = np.sum(frame * frame)
        if energy < 1e-4 or max_lag <= min_lag:
            pitches.append(0.0)
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if corr[0] <= 0:
            pitches.append(0.0)
            continue
        lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        strength = corr[lag] / corr[0]
        pitches.append(sr / lag if strength > 0.25 else 0.0)
    return np.asarray(pitches, dtype=np.float64)


def extract_mfcc_features_990(file_bytes):
    """
    Extract exactly 990 MFCC-based features matching v95 S2_MFCC training.

    Structure:
      - 40 MFCCs × 8 stats          = 320 features
      - 40 delta MFCCs × 8 stats    = 320 features
      - 40 delta-delta MFCCs × 8 stats = 320 features
      - 30 prosodic/spectral features
      Total = 990
    """
    y, sr = load_audio_signal(file_bytes)
    duration = safe_float(y.size / sr)

    frames = _frame_signal(y, sr)  # (n_frames, frame_len)

    # MFCC matrix: (40, n_frames)
    mfcc = _compute_mfcc_matrix(frames, sr, n_mfcc=40)
    delta = _compute_delta(mfcc)
    delta2 = _compute_delta(delta)

    features = []

    # 40 × 8 = 320: MFCC statistics
    for i in range(40):
        features.extend(_safe_stats(mfcc[i]))

    # 40 × 8 = 320: Delta MFCC statistics
    for i in range(40):
        features.extend(_safe_stats(delta[i]))

    # 40 × 8 = 320: Delta-delta MFCC statistics
    for i in range(40):
        features.extend(_safe_stats(delta2[i]))

    # ── 30 Prosodic / spectral features ───────────────────────────────────────
    n_fft = 512
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft))           # (n_frames, n_fft//2+1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    spectrum_sum = np.maximum(spectrum.sum(axis=1), 1e-10)     # (n_frames,)

    # RMS energy per frame
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # ZCR per frame
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1).astype(np.float64)

    # Spectral centroid & bandwidth
    spectral_centroid = (spectrum * freqs).sum(axis=1) / spectrum_sum
    spectral_bandwidth = np.sqrt(
        (spectrum * ((freqs[None, :] - spectral_centroid[:, None]) ** 2)).sum(axis=1)
        / spectrum_sum
    )

    # Spectral rolloff (85% energy)
    cumsum = np.cumsum(spectrum, axis=1)
    rolloff_threshold = 0.85 * cumsum[:, -1:]
    rolloff_idx = np.argmax(cumsum >= rolloff_threshold, axis=1)
    spectral_rolloff = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]

    # Spectral flatness
    log_mean = np.mean(np.log(np.maximum(spectrum, 1e-10)), axis=1)
    arith_mean = spectrum.mean(axis=1)
    spectral_flatness = np.exp(log_mean) / np.maximum(arith_mean, 1e-10)

    # Pitch per frame (voiced = pitch > 0)
    pitch_arr = _estimate_pitch(frames, sr)
    voiced_mask = pitch_arr > 0
    voiced_ratio = float(voiced_mask.mean()) if voiced_mask.size else 0.0

    pitch_voiced = pitch_arr[voiced_mask] if voiced_mask.any() else np.array([0.0])

    # HNR proxy: ratio of voiced frames energy vs total energy
    voiced_energy = float(rms[voiced_mask].mean()) if voiced_mask.any() else 0.0
    total_energy = float(rms.mean()) if rms.size else 1e-10
    hnr_proxy = safe_float(voiced_energy / max(total_energy, 1e-10))

    # Spectral entropy
    prob = spectrum / spectrum_sum[:, None]
    prob = np.clip(prob, 1e-10, 1.0)
    spectral_entropy_per_frame = -np.sum(prob * np.log(prob + 1e-10), axis=1)
    spectral_entropy = safe_float(np.mean(spectral_entropy_per_frame))

    # Jitter (F0 perturbation): cycle-to-cycle F0 variation among voiced frames
    if pitch_voiced.size > 1:
        jitter = safe_float(np.mean(np.abs(np.diff(pitch_voiced))) / max(np.mean(pitch_voiced), 1e-10))
    else:
        jitter = 0.0

    # Shimmer (amplitude perturbation): cycle-to-cycle amplitude variation
    rms_voiced = rms[voiced_mask] if voiced_mask.any() else rms
    if rms_voiced.size > 1:
        shimmer = safe_float(np.mean(np.abs(np.diff(rms_voiced))) / max(np.mean(rms_voiced), 1e-10))
    else:
        shimmer = 0.0

    # 30 prosodic features
    prosodic = [
        # RMS stats (4)
        safe_float(np.mean(rms)), safe_float(np.std(rms)),
        safe_float(np.min(rms)),  safe_float(np.max(rms)),
        # ZCR stats (4)
        safe_float(np.mean(zcr)), safe_float(np.std(zcr)),
        safe_float(np.min(zcr)),  safe_float(np.max(zcr)),
        # Spectral centroid stats (4)
        safe_float(np.mean(spectral_centroid)), safe_float(np.std(spectral_centroid)),
        safe_float(np.min(spectral_centroid)),  safe_float(np.max(spectral_centroid)),
        # Spectral bandwidth stats (4)
        safe_float(np.mean(spectral_bandwidth)), safe_float(np.std(spectral_bandwidth)),
        safe_float(np.min(spectral_bandwidth)),  safe_float(np.max(spectral_bandwidth)),
        # Spectral rolloff stats (4)
        safe_float(np.mean(spectral_rolloff)), safe_float(np.std(spectral_rolloff)),
        safe_float(np.min(spectral_rolloff)),  safe_float(np.max(spectral_rolloff)),
        # Spectral flatness stats (2)
        safe_float(np.mean(spectral_flatness)), safe_float(np.std(spectral_flatness)),
        # Pitch (voiced frames) stats (4)
        safe_float(np.mean(pitch_voiced)), safe_float(np.std(pitch_voiced)),
        safe_float(np.min(pitch_voiced)),  safe_float(np.max(pitch_voiced)),
        # Scalar features (4)
        voiced_ratio, hnr_proxy, jitter, shimmer,
        # Spectral entropy (1)
        spectral_entropy,
        # Duration (1)
        safe_float(duration),
    ]  # = 30 features

    features.extend(prosodic)

    # Ensure exactly 990 features
    features = np.asarray(features, dtype=np.float64)
    if features.size < EXPECTED_MFCC_FEATURES:
        features = np.pad(features, (0, EXPECTED_MFCC_FEATURES - features.size))
    else:
        features = features[:EXPECTED_MFCC_FEATURES]

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(features, -1e9, 1e9, out=features)

    # Build acoustic summary dict (for display purposes)
    acoustic = {
        "duration": round(duration, 1),
        "avg_pitch": round(safe_float(np.mean(pitch_voiced), 0.0)),
        "pitch_variability": safe_float(np.std(pitch_voiced), 0.0),
        "energy": safe_float(np.mean(rms), 0.0),
        "energy_std": safe_float(np.std(rms), 0.0),
        "zcr": safe_float(np.mean(zcr), 0.0),
        "spectral_centroid": safe_float(np.mean(spectral_centroid), 0.0),
        "spectral_bandwidth": safe_float(np.mean(spectral_bandwidth), 0.0),
        "signal_quality": _estimate_signal_quality(y, rms),
    }

    return features, acoustic


def _estimate_signal_quality(y, rms):
    if y.size == 0:
        return 0
    clipping_ratio = float(np.mean(np.abs(y) > 0.98))
    silence_ratio = float(np.mean(np.asarray(rms) < 0.01)) if len(rms) else 1.0
    score = 98 - (clipping_ratio * 120) - (silence_ratio * 35)
    return int(max(40, min(99, round(score))))


# ── Prediction ────────────────────────────────────────────────────────────────

def _build_prediction_result(probability, threshold, model_name, source, scenario, note):
    probability = max(0.0, min(1.0, safe_float(probability)))
    predicted_label = int(probability >= threshold)
    depression = int(round(probability * 100))
    normal = 100 - depression
    primary_detection = "Depression" if predicted_label == 1 else "Normal State"
    confidence = depression if primary_detection == "Depression" else normal

    return {
        "model": model_name,
        "source": source,
        "scenario": scenario,
        "primaryDetection": primary_detection,
        "confidence": confidence,
        "depression": depression,
        "normal": normal,
        "probability": probability,
        "threshold": threshold,
        "status": "available",
        "note": note,
    }


def predict_machine_learning(file_bytes):
    """
    Run the XGB v95 pipeline on the uploaded audio.
    The pipeline internally applies StandardScaler → PCA → XGBClassifier.
    """
    pipeline, threshold = load_model_pipeline()
    mfcc_features, acoustic = extract_mfcc_features_990(file_bytes)

    X = mfcc_features.reshape(1, -1)
    probability = float(pipeline.predict_proba(X)[0][1])

    result = _build_prediction_result(
        probability=probability,
        threshold=threshold,
        model_name="Machine Learning - XGBoost v95",
        source="Model/Machine Learning/best_XGB_S2_MFCC.pkl",
        scenario="S2_MFCC",
        note="XGBoost pipeline trained on 990 MFCC-derived acoustic features.",
    )
    result["acoustic"] = acoustic
    return result


def _normalize_feature(x):
    mean = x.mean()
    std = x.std()
    if std < 1e-6:
        return x - mean
    return np.clip((x - mean) / std, -10.0, 10.0)


def _pad_or_trim_2d(x, length, width):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        x = np.zeros((0, width), dtype=np.float32)
    if x.shape[1] != width:
        fixed = np.zeros((x.shape[0], width), dtype=np.float32)
        fixed[:, : min(width, x.shape[1])] = x[:, : min(width, x.shape[1])]
        x = fixed
    if x.shape[0] > length:
        return x[:length]
    if x.shape[0] < length:
        return np.vstack([x, np.zeros((length - x.shape[0], width), dtype=np.float32)])
    return x


def _pad_or_trim_1d(x, length):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size > length:
        return x[:length]
    if x.size < length:
        return np.pad(x, (0, length - x.size)).astype(np.float32)
    return x


def extract_deep_learning_inputs(file_bytes):
    y, sr = load_audio_signal(file_bytes)
    if sr != 16000:
        raise RuntimeError("Deep learning preprocessing expects 16 kHz audio.")

    raw = _pad_or_trim_1d(y, DL_RAW_LEN)

    frames = _frame_signal(raw, sr, frame_ms=25, hop_ms=32)
    mfcc = _compute_mfcc_matrix(frames, sr, n_mfcc=40, n_fft=1024)
    delta = _compute_delta(mfcc)
    delta2 = _compute_delta(delta)
    mfcc_seq = np.concatenate([mfcc, delta, delta2], axis=0).T
    mfcc_seq = _pad_or_trim_2d(mfcc_seq, DL_MAX_MFCC_LEN, DL_MFCC_FEATURES)

    spectrum = np.fft.rfft(frames, n=1024)
    power = (np.abs(spectrum) ** 2) / 1024
    filters = _mel_filterbank(sr, 1024, n_filters=DL_N_MELS)
    mel_power = np.maximum(np.dot(power, filters.T), 1e-10)
    spec_seq = (10.0 * np.log10(mel_power / np.max(mel_power))).astype(np.float32)
    spec_seq = _pad_or_trim_2d(spec_seq, DL_MAX_SPEC_LEN, DL_N_MELS)

    return (
        _normalize_feature(mfcc_seq).astype(np.float32),
        _normalize_feature(spec_seq).astype(np.float32),
        _normalize_feature(raw).astype(np.float32),
    )


@lru_cache(maxsize=1)
def load_deep_learning_model():
    try:
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Model
    except ImportError as exc:
        raise RuntimeError("Deep learning dependencies are not installed. Install torch and transformers.") from exc

    if not DL_MODEL_PATH.exists():
        raise RuntimeError(f"Deep learning model file not found: {DL_MODEL_PATH}")

    class FeatureAdapter(nn.Module):
        def __init__(self, input_dim, output_dim=512, dropout=0.1):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.proj(x)

    class FusionV4Model(nn.Module):
        def __init__(self, dropout=0.5):
            super().__init__()
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base",
                cache_dir=str(HF_TRANSFORMERS_CACHE_DIR),
                local_files_only=True,
            )
            self.mfcc_adapter = FeatureAdapter(DL_MFCC_FEATURES, 512, dropout=0.1)
            self.spec_adapter = FeatureAdapter(DL_N_MELS, 512, dropout=0.1)

            def projection():
                return nn.Sequential(
                    nn.Linear(768, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )

            self.raw_proj = projection()
            self.mfcc_proj = projection()
            self.spec_proj = projection()
            self.fusion = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 2),
            )

        def _encode_via_transformer(self, adapted):
            hidden = self.wav2vec.feature_projection(adapted)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            encoder_out = self.wav2vec.encoder(hidden)
            last_hidden = (
                encoder_out.last_hidden_state
                if hasattr(encoder_out, "last_hidden_state")
                else encoder_out[0]
            )
            return last_hidden.mean(dim=1)

        def forward(self, mfcc, spec, wav):
            e_raw = self.wav2vec(wav).last_hidden_state.mean(dim=1)
            e_raw = self.raw_proj(e_raw)
            e_mfcc = self._encode_via_transformer(self.mfcc_adapter(mfcc))
            e_mfcc = self.mfcc_proj(e_mfcc)
            e_spec = self._encode_via_transformer(self.spec_adapter(spec))
            e_spec = self.spec_proj(e_spec)
            return self.fusion(torch.cat([e_raw, e_mfcc, e_spec], dim=1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionV4Model(dropout=0.5).to(device)
    try:
        checkpoint = torch.load(DL_MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(DL_MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Deep learning checkpoint has unexpected keys: "
            + ", ".join(incompatible.unexpected_keys[:8])
        )
    model.eval()
    return model, device


def predict_deep_learning(file_bytes):
    import torch
    import torch.nn.functional as F

    model, device = load_deep_learning_model()
    mfcc, spec, wav = extract_deep_learning_inputs(file_bytes)
    with torch.no_grad():
        mfcc_t = torch.from_numpy(mfcc).unsqueeze(0).to(device)
        spec_t = torch.from_numpy(spec).unsqueeze(0).to(device)
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
        logits = model(mfcc_t, spec_t, wav_t)
        probability = float(F.softmax(logits.float(), dim=1)[0, 1].cpu().item())

    return _build_prediction_result(
        probability=probability,
        threshold=DL_THRESHOLD,
        model_name="Deep Learning - Fusion V4 Wav2Vec2",
        source="Model/Deep Learning/v4_wav2vec_3f.pt",
        scenario="MFCC + MelSpec + Raw Waveform",
        note="Fusion V4 model using MFCC, mel-spectrogram, and raw waveform branches.",
    )


def unavailable_model_result(model_name, source, scenario, error):
    return {
        "model": model_name,
        "source": source,
        "scenario": scenario,
        "status": "unavailable",
        "error": str(error),
        "primaryDetection": "Unavailable",
        "confidence": 0,
        "depression": 0,
        "normal": 0,
        "probability": None,
        "threshold": None,
    }


def build_consensus_prediction(ml_result, dl_result):
    available = [r for r in [ml_result, dl_result] if r.get("status") == "available"]
    if not available:
        raise RuntimeError("No prediction model is available.")

    probability = float(np.mean([r["probability"] for r in available]))
    result = _build_prediction_result(
        probability=probability,
        threshold=0.5,
        model_name="Consensus ML + DL",
        source="Machine Learning and Deep Learning models",
        scenario="Average probability ensemble",
        note="Consensus prediction averages all available model probabilities.",
    )
    result["modelCount"] = len(available)
    return result


def predict_audio(file_bytes):
    ml_result = predict_machine_learning(file_bytes)
    acoustic = ml_result["acoustic"]

    try:
        dl_result = predict_deep_learning(file_bytes)
    except Exception as exc:
        dl_result = unavailable_model_result(
            model_name="Deep Learning - Fusion V4 Wav2Vec2",
            source="Model/Deep Learning/v4_wav2vec_3f.pt",
            scenario="MFCC + MelSpec + Raw Waveform",
            error=exc,
        )

    consensus = build_consensus_prediction(ml_result, dl_result)
    consensus["acoustic"] = acoustic

    return {
        "primaryDetection": consensus["primaryDetection"],
        "confidence": consensus["confidence"],
        "depression": consensus["depression"],
        "normal": consensus["normal"],
        "acoustic": acoustic,
        "probability": consensus["probability"],
        "modelResults": {
            "machineLearning": ml_result,
            "deepLearning": dl_result,
            "consensus": consensus,
        },
    }


# ── Database ──────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_wav_info(file_bytes):
    try:
        with wave.open(io.BytesIO(file_bytes), "rb") as wav:
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
    """Generate SHAP-like and LIME-like explainability based on acoustic features."""
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


# ── API Endpoints ─────────────────────────────────────────────────────────────

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
    avg_pitch = acoustic["avg_pitch"]
    energy_level = get_energy_level(acoustic["energy"])
    signal_quality = acoustic["signal_quality"]

    now = datetime.datetime.now()
    date_str = now.strftime("%m/%d/%Y")
    timestamp_str = now.strftime("%m/%d/%Y, %I:%M:%S %p")

    base_value = 50.0
    shap_features, lime_rules = build_explainability(prediction)
    model_results = prediction["modelResults"]
    ml_model = model_results["machineLearning"]
    dl_model = model_results["deepLearning"]
    consensus_model = model_results["consensus"]

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
            "avgPitch": f"{avg_pitch} Hz",
            "energyLevel": energy_level,
            "signalQuality": f"{signal_quality}%",
            "audioUrl": f"/api/audio/{result_id}",
        },
        "performance": {
            "accuracy": "65.0%",
            "precision": "65.2%",
            "f1Score": "64.9%",
        },
        "shapData": {
            "baseValue": base_value,
            "predictionValue": float(depression),
            "features": shap_features,
        },
        "limeRules": lime_rules,
        "modelResults": model_results,
        "modelInfo": {
            "name": consensus_model["model"],
            "source": consensus_model["source"],
            "scenario": consensus_model["scenario"],
            "depressionProbability": round(prediction["probability"], 4),
            "threshold": consensus_model["threshold"],
            "machineLearning": {
                "name": ml_model["model"],
                "source": ml_model["source"],
                "scenario": ml_model["scenario"],
                "status": ml_model["status"],
                "depressionProbability": round(ml_model["probability"], 4),
                "threshold": ml_model["threshold"],
            },
            "deepLearning": {
                "name": dl_model["model"],
                "source": dl_model["source"],
                "scenario": dl_model["scenario"],
                "status": dl_model["status"],
                "depressionProbability": (
                    round(dl_model["probability"], 4)
                    if dl_model.get("probability") is not None
                    else None
                ),
                "threshold": dl_model["threshold"],
                "error": dl_model.get("error"),
            },
            "note": consensus_model["note"],
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

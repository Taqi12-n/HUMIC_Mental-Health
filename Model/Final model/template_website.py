"""
template_website.py
===================
Template Inferensi untuk Web Developer — Deteksi Depresi via Audio

Model   : ML_Fusion_RF (Random Forest, Fusion Feature, Test F1=0.70)
Model file : appletoapplefinal/models/ML/ML_Fusion_RF/best_model.pkl
Scenario: Fusion = MelSpec(128) + MFCC(120) + Wav2Vec(768) → 1016-dim

Cara pakai:
    python template_website.py --audio path/ke/audio.wav
    python template_website.py --audio path/ke/audio.mp3

Atau sebagai modul:
    from template_website import predict_depression
    result = predict_depression("path/to/audio.wav")
    print(result)

Output (JSON):
    {
        "participant_id": "unknown",
        "model_used": "ML_Fusion_RF",
        "total_segments": 5,
        "segment_votes": {"NORMAL": 2, "DEPRESI": 3},
        "probability": {"NORMAL": 0.40, "DEPRESI": 0.60},
        "prediction": "DEPRESI",
        "confidence": "60.00%",
        "segment_detail": [...],
        "note": "Patient-level majority voting dari 5 segmen audio"
    }

Preprocessing Pipeline (sama persis dengan training):
    1. Load audio → resample ke 16kHz, mono
    2. VAD — potong silence, ambil bagian ada suara
    3. Bagi jadi segmen ~20-30 detik (merge 5 utterance pendek)
    4. Per segmen:
       a. Ekstrak MelSpec (128,T) → mean axis=1 → 128-dim
       b. Ekstrak MFCC 40+delta+delta2 (T,120) → mean axis=0 → 120-dim
       c. Ekstrak Wav2Vec2 (T,768) → mean axis=0 → 768-dim
    5. Concat [MelSpec | MFCC | Wav2Vec] → 1016-dim per segmen
    6. Scaler transform (sudah embedded di Pipeline sklearn)
    7. RF predict_proba → threshold 0.37 → majority vote

Dependencies:
    pip install numpy librosa joblib scikit-learn transformers torch
    (torch diperlukan untuk Wav2Vec, bisa CPU)

Notes untuk Web Developer:
    - Ganti MODEL_PATH sesuai lokasi server
    - Audio input bisa WAV atau MP3 (librosa menangani keduanya)
    - Untuk production: tambahkan validasi durasi minimum audio (~10 detik)
    - Jika tidak punya GPU: set DEVICE = 'cpu' (lebih lambat tapi berjalan)
    - Wav2Vec butuh internet pertama kali (download ~360MB model pretrained)
      Setelah itu di-cache otomatis oleh HuggingFace
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI — Sesuaikan dengan path di server kalian
# ============================================================

# Path ke model pkl (ganti sesuai deployment path)
MODEL_PATH = Path(__file__).parent / "models" / "ML" / "ML_Fusion_RF" / "best_model.pkl"

# Wav2Vec2 pretrained model dari HuggingFace
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base"

# Audio config (harus sama persis dengan training)
SAMPLE_RATE     = 16000          # 16kHz
MIN_SEG_DUR     = 0.5            # Skip segmen < 0.5 detik
MERGE_N         = 5              # Merge tiap 5 ucapan pendek → ~20-30 detik
MAX_AUDIO_SECS  = 15             # Truncate audio > 15 detik sebelum Wav2Vec
MAX_W2V_FRAMES  = 249            # Fixed output frames Wav2Vec

# Feature dimensions (jangan diubah)
MELSPEC_DIM     = 128
MFCC_DIM        = 120
W2V_DIM         = 768
FUSION_DIM      = MELSPEC_DIM + MFCC_DIM + W2V_DIM  # 1016

# MelSpec / MFCC parameters (sama dengan reextract_features_v2.py)
N_MELS      = 128
N_MFCC      = 40
HOP_LENGTH  = 512
N_FFT       = 1024

CLASS_NAMES = ["NORMAL", "DEPRESI"]

# Device untuk Wav2Vec (gunakan 'cpu' jika tidak ada GPU)
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'


# ============================================================
# 1. LOAD MODEL
# ============================================================

def load_model(model_path: Path) -> dict:
    """
    Load ML_Fusion_RF model dari file .pkl.

    Struktur pkl yang disimpan oleh tradisional_mlv101.py:
        {
            'pipeline'       : sklearn.Pipeline (StandardScaler + RandomForestClassifier),
            'threshold'      : float  — optimal dari OOF sweep (misal 0.37)
            'scenario'       : 'Fusion',
            'model'          : 'RF',
            'cv_mean_f1'     : float,
            'test_f1'        : float,
            'class_names'    : ['NORMAL', 'DEPRESI'],
            ...
        }

    PENTING: Scaler sudah embedded di dalam 'pipeline' (sklearn Pipeline).
    Tidak perlu load scaler terpisah.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}\n"
            f"Pastikan path ke best_model.pkl sudah benar."
        )
    data = joblib.load(model_path)
    print(f"[MODEL] Loaded: {data.get('model','?')} | "
          f"Scenario: {data.get('scenario','?')} | "
          f"Threshold: {data.get('threshold', 0.5):.2f} | "
          f"Test F1: {data.get('test_f1', '?')}")
    return data


# ============================================================
# 2. AUDIO LOADING & VAD SEGMENTATION
# ============================================================

def load_audio(audio_path: str) -> np.ndarray:
    """
    Load audio file (WAV/MP3) dan resample ke 16kHz mono.
    Librosa mendukung WAV, MP3, FLAC, OGG, dll.
    """
    audio_path = str(audio_path)
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / SAMPLE_RATE
    print(f"[AUDIO] Loaded: {Path(audio_path).name} | "
          f"Duration: {duration:.1f}s | SR: {SAMPLE_RATE}Hz")
    return y.astype(np.float32)


def vad_segmentation(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list:
    """
    Voice Activity Detection (VAD) — potong audio menjadi segmen ucapan.

    Strategi (mengikuti cut_participant_segments.py):
        1. Hitung RMS energy per frame
        2. Ambil frame di atas threshold (ada suara)
        3. Gabungkan frame yang berdekatan → utterance-utterance pendek
        4. Merge tiap MERGE_N=5 utterance → 1 segmen panjang (~20-30 detik)

    Note: Pada training, kita bisa baca TRANSCRIPT.csv untuk tau
    persis kapan partisipan berbicara. Untuk inference dari web,
    kita pakai energy-based VAD sebagai pengganti yang cukup baik.

    Note untuk Web Developer:
        Untuk production yang lebih robust, bisa pakai:
        - webrtcvad : pip install webrtcvad
        - silero-vad: torch.hub.load('snakers4/silero-vad', 'silero_vad')

    Returns:
        list of np.ndarray — segmen audio, masing-masing ~20-30 detik
    """
    hop = 512
    frame_length = 1024

    # Hitung RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
    threshold = rms.max() * 0.10   # 10% dari RMS max

    gap_tolerance_frames = int(0.5 * sr / hop)  # 0.5 detik gap → masih 1 utterance

    # Temukan utterance (blok berurutan di atas threshold)
    utterances = []
    in_speech  = False
    start_frame = 0
    gap_count   = 0

    for i, v in enumerate(rms > threshold):
        if v:
            if not in_speech:
                start_frame = i
                in_speech   = True
            gap_count = 0
        else:
            if in_speech:
                gap_count += 1
                if gap_count > gap_tolerance_frames:
                    end_frame    = i - gap_count
                    start_sample = start_frame * hop
                    end_sample   = min(end_frame * hop, len(audio))
                    dur = (end_sample - start_sample) / sr
                    if dur >= MIN_SEG_DUR:
                        utterances.append(audio[start_sample:end_sample])
                    in_speech = False
                    gap_count = 0

    # Handle utterance terakhir
    if in_speech:
        start_sample = start_frame * hop
        end_sample   = len(audio)
        dur = (end_sample - start_sample) / sr
        if dur >= MIN_SEG_DUR:
            utterances.append(audio[start_sample:end_sample])

    # Fallback: jika tidak ada utterance terdeteksi → pakai seluruh audio
    if not utterances:
        print("[VAD] Tidak ada segmen terdeteksi, gunakan seluruh audio sebagai 1 segmen")
        return [audio]

    # Merge tiap MERGE_N utterance → 1 segmen panjang
    silence  = np.zeros(int(0.1 * sr), dtype=np.float32)   # 0.1 detik silence antar merge
    segments = []
    for i in range(0, len(utterances), MERGE_N):
        chunk  = utterances[i:i + MERGE_N]
        merged = silence.copy()
        for utt in chunk:
            merged = np.concatenate([merged, utt, silence])
        if len(merged) / sr >= MIN_SEG_DUR:
            segments.append(merged)

    total_dur = sum(len(s) / sr for s in segments)
    print(f"[VAD] Utterances: {len(utterances)} | "
          f"Segments setelah merge ({MERGE_N}/seg): {len(segments)} | "
          f"Total speech: {total_dur:.1f}s")

    return segments


# ============================================================
# 3. FEATURE EXTRACTION PER SEGMEN
# ============================================================

def extract_melspec(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Ekstrak Mel-Spectrogram (log-scaled).
    Output shape: (N_MELS=128, T)
    Identik dengan reextract_features_v2.py → extract_melspec()
    """
    mel    = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)   # (128, T)


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Ekstrak MFCC + Delta + Delta².
    Output shape: (T, 120)  — 40 MFCC × 3
    Identik dengan reextract_features_v2.py → extract_mfcc()
    """
    mfcc    = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])   # (120, T)
    return features.T.astype(np.float32)           # (T, 120)


def load_wav2vec_model(model_name: str = WAV2VEC_MODEL_NAME, device: str = DEVICE):
    """
    Load Wav2Vec2 model dan processor.
    Panggil SEKALI saat startup, reuse untuk semua request.
    Download otomatis dari HuggingFace (~360MB, tersimpan di cache).
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    print(f"[WAV2VEC] Loading {model_name} on {device}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[WAV2VEC] Ready! ({param_count:.1f}M params, device={device})")
    return model, processor


def extract_wav2vec(
    audio: np.ndarray,
    w2v_model,
    w2v_processor,
    device: str = DEVICE
) -> Optional[np.ndarray]:
    """
    Ekstrak Wav2Vec2 embeddings.
    Output shape: (MAX_W2V_FRAMES=249, 768)
    Identik dengan extract_wav2vec_features.py → infer_single()
    """
    try:
        import torch

        # Truncate audio > MAX_AUDIO_SECS detik (cegah OOM)
        max_samples = SAMPLE_RATE * MAX_AUDIO_SECS
        audio_trunc = audio[:max_samples] if len(audio) > max_samples else audio

        # Preprocess → input tensor
        inputs = w2v_processor(
            audio_trunc, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=False
        )
        input_values = inputs.input_values.to(device)

        # Inference
        with torch.no_grad():
            out = w2v_model(input_values)
            h   = out.last_hidden_state.squeeze(0).cpu().numpy()  # (T, 768)

        # Pad / truncate ke fixed length MAX_W2V_FRAMES=249
        T = h.shape[0]
        if T >= MAX_W2V_FRAMES:
            h = h[:MAX_W2V_FRAMES]
        else:
            h = np.pad(h, ((0, MAX_W2V_FRAMES - T), (0, 0)))

        return h.astype(np.float32)   # (249, 768)

    except Exception as e:
        print(f"  [WAV2VEC ERROR] {e}")
        return None
    finally:
        if device == 'cuda':
            import torch
            torch.cuda.empty_cache()


# ============================================================
# 4. MEAN POOLING (identik dengan tradisional_mlv101.py)
# ============================================================

def mean_pool(arr: np.ndarray, feature_type: str) -> np.ndarray:
    """
    Konversi 2D feature matrix → 1D mean vector.

    MelSpec (128, T) → mean axis=1 → (128,)
    MFCC    (T, 120) → mean axis=0 → (120,)
    Wav2Vec (T, 768) → mean axis=0 → (768,)

    Normalisasi Z-score per segmen sebelum pooling
    (identik dengan _mean_pool() di tradisional_mlv101.py)
    """
    feat = arr.astype(np.float32)
    feat = (feat - feat.mean()) / (feat.std() + 1e-8)   # z-score per segmen

    if feature_type == 'melspec':
        return feat.mean(axis=1)   # (128, T) → (128,)
    else:
        return feat.mean(axis=0)   # (T, F)   → (F,)


# ============================================================
# 5. INFERENCE UTAMA
# ============================================================

def predict_depression(
    audio_path: str,
    model_path: Path = MODEL_PATH,
    participant_id: str = "unknown",
    w2v_model=None,
    w2v_processor=None,
    enable_xai: bool = True,
    shap_explainer=None,
) -> dict:
    """
    Fungsi utama inferensi: audio file → JSON prediksi.

    Args:
        audio_path     : Path ke file audio (.wav / .mp3)
        model_path     : Path ke best_model.pkl
        participant_id : ID pasien (opsional, untuk tracking di log)
        w2v_model      : Wav2Vec2Model yang sudah di-load (None = load otomatis)
        w2v_processor  : Wav2Vec2Processor yang sudah di-load (None = load otomatis)

    Returns:
        dict — JSON-serializable, berisi prediction, confidence, segment_votes, dll.
    """
    print(f"\n{'='*60}")
    print(f"  INFERENCE: {Path(audio_path).name}")
    print(f"{'='*60}")

    # ── Load model ────────────────────────────────────────────
    model_data = load_model(model_path)
    pipeline   = model_data['pipeline']        # sklearn Pipeline (Scaler + RF)
    threshold  = model_data.get('threshold', 0.5)

    # ── Load Wav2Vec (lazy load jika belum ada) ───────────────
    if w2v_model is None or w2v_processor is None:
        w2v_model, w2v_processor = load_wav2vec_model(device=DEVICE)

    # ── Load & segmentasi audio ───────────────────────────────
    audio    = load_audio(audio_path)
    segments = vad_segmentation(audio)

    if not segments:
        return {
            "error": "Tidak ada segmen audio yang terdeteksi.",
            "participant_id": participant_id
        }

    # ── Ekstrak fitur per segmen → fusion vector ──────────────
    fusion_vectors = []
    segment_detail = []

    for i, seg in enumerate(segments):
        seg_dur = len(seg) / SAMPLE_RATE
        print(f"[SEG {i+1:02d}/{len(segments):02d}] duration={seg_dur:.1f}s")

        # a. MelSpec: (128, T) → mean pool → (128,)
        mel     = extract_melspec(seg)
        mel_vec = mean_pool(mel, 'melspec')

        # b. MFCC+delta+delta2: (T, 120) → mean pool → (120,)
        mfcc     = extract_mfcc(seg)
        mfcc_vec = mean_pool(mfcc, 'mfcc')

        # c. Wav2Vec2: (249, 768) → mean pool → (768,)
        w2v = extract_wav2vec(seg, w2v_model, w2v_processor, device=DEVICE)
        if w2v is None:
            print(f"  [WARN] Wav2Vec gagal untuk segmen {i+1}, segmen ini di-skip")
            continue
        w2v_vec = mean_pool(w2v, 'wav2vec')

        # d. Fusion concat → (1016,)
        fusion = np.concatenate([mel_vec, mfcc_vec, w2v_vec]).astype(np.float64)
        fusion_vectors.append(fusion)
        segment_detail.append({
            "segment_index": i + 1,
            "duration_sec":  round(seg_dur, 2),
            "feature_dim":   int(fusion.shape[0]),
        })

    if not fusion_vectors:
        return {
            "error": "Semua segmen gagal diekstrak fiturnya (Wav2Vec error).",
            "participant_id": participant_id
        }

    # ── Prediksi per segmen ───────────────────────────────────
    X      = np.vstack(fusion_vectors)           # (N_segs, 1016)
    probs  = pipeline.predict_proba(X)[:, 1]     # P(DEPRESI) per segmen
    preds  = (probs >= threshold).astype(int)    # 0=NORMAL, 1=DEPRESI

    for i, (pred, prob) in enumerate(zip(preds, probs)):
        segment_detail[i]["pred_label"]   = CLASS_NAMES[int(pred)]
        segment_detail[i]["prob_depresi"] = round(float(prob), 4)
        segment_detail[i]["prob_normal"]  = round(float(1 - prob), 4)

    # ── Majority voting → 1 keputusan final per pasien ────────
    n_depresi   = int(preds.sum())
    n_normal    = len(preds) - n_depresi
    final_pred  = int(np.bincount(preds).argmax())

    # Probabilitas final = rata-rata P(DEPRESI) semua segmen
    mean_prob_depresi = float(probs.mean())
    mean_prob_normal  = float(1 - mean_prob_depresi)
    confidence_pct    = max(mean_prob_depresi, mean_prob_normal) * 100

    result = {
        "participant_id":  participant_id,
        "model_used":      "ML_Fusion_RF",
        "audio_file":      str(Path(audio_path).name),
        "total_segments":  int(len(preds)),
        "threshold_used":  round(float(threshold), 2),
        "segment_votes": {
            "NORMAL":  int(n_normal),
            "DEPRESI": int(n_depresi),
        },
        "probability": {
            "NORMAL":  round(mean_prob_normal, 4),
            "DEPRESI": round(mean_prob_depresi, 4),
        },
        "prediction":      CLASS_NAMES[final_pred],
        "confidence":      f"{confidence_pct:.2f}%",
        "segment_detail":  segment_detail,
        "note": (
            f"Patient-level majority voting dari {len(preds)} segmen audio. "
            f"Threshold: {threshold:.2f}. "
            f"Votes: NORMAL={n_normal}, DEPRESI={n_depresi}."
        )
    }

    print(f"\n{'='*60}")
    print(f"  HASIL    : {result['prediction']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Votes    : NORMAL={n_normal}, DEPRESI={n_depresi} / {len(preds)} segmen")
    print(f"{'='*60}\n")

    # ── XAI: Tambah SHAP explanation (opsional) ───────────────
    if enable_xai and len(fusion_vectors) > 0:
        print("[XAI] Menghitung SHAP explanation...")
        shap_result = explain_prediction_shap(
            pipeline=pipeline,
            X_raw=X,
            shap_explainer=shap_explainer,
        )
        result["shap_explanation"] = shap_result

    return result


# ============================================================
# 6. XAI — SHAP EXPLANATION (4 LAYERS)
# ============================================================

# Feature index boundaries
_FEAT_IDX = {
    'melspec':     (0,   128),
    'mfcc_base':   (128, 168),
    'mfcc_delta':  (168, 208),
    'mfcc_delta2': (208, 248),
    'wav2vec':     (248, 1016),
}
_MELSPEC_SUB = {
    'Low freq (band 1-32)':    (0,   32),
    'Mid freq (band 33-80)':   (32,  80),
    'High freq (band 81-128)': (80,  128),
}
_MFCC_SUB = {
    'MFCC Base (C1-C40)': (128, 168),
    'MFCC Delta':         (168, 208),
    'MFCC Delta2':        (208, 248),
}


def load_shap_explainer(pipeline):
    """
    Buat SHAP TreeExplainer dari RF dalam pipeline.
    Panggil SEKALI saat startup, reuse untuk semua request.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")
    rf = pipeline.named_steps['clf']
    explainer = shap.TreeExplainer(rf)
    print("[XAI] SHAP TreeExplainer loaded")
    return explainer


def explain_prediction_shap(
    pipeline,
    X_raw: np.ndarray,
    shap_explainer=None,
    top_n_waterfall: int = 15,
) -> dict:
    """
    Hitung SHAP explanation untuk batch segmen audio.
    Mengembalikan 4 layer siap dikonsumsi website.

    Args:
        pipeline       : sklearn Pipeline (sudah di-load)
        X_raw          : fusion features SEBELUM scaling, shape (N, 1016)
        shap_explainer : SHAP TreeExplainer (opsional, buat jika None)
        top_n_waterfall: jumlah fitur teratas di waterfall

    Returns:
        dict dengan layer1/layer2/layer3/layer4
    """
    try:
        import shap as _shap
    except ImportError:
        return {"error": "shap tidak terinstall. pip install shap"}

    scaler = pipeline.named_steps['scaler']
    X_scaled = scaler.transform(X_raw)   # (N, 1016)

    # Buat explainer jika belum ada
    if shap_explainer is None:
        shap_explainer = load_shap_explainer(pipeline)

    # Hitung SHAP — rata-rata semua segmen audio ini
    shap_values = shap_explainer.shap_values(X_scaled, check_additivity=False)
    shap_depresi = shap_values[1]          # (N, 1016), untuk kelas DEPRESI
    expected_val = float(shap_explainer.expected_value[1])

    # ── Layer 1: Group contribution ───────────────────────────
    abs_shap = np.abs(shap_depresi)       # (N, 1016)
    total    = abs_shap.sum()
    layer1   = {}
    for grp, (s, e) in [('MelSpec',(0,128)), ('MFCC',(128,248)), ('Wav2Vec',(248,1016))]:
        grp_abs  = float(abs_shap[:, s:e].sum())
        mean_sv  = float(shap_depresi[:, s:e].mean())
        layer1[grp] = {
            'contribution_pct': round(grp_abs / total * 100, 2) if total > 0 else 0.0,
            'mean_shap':        round(mean_sv, 6),
            'direction':        'DEPRESI' if mean_sv >= 0 else 'NORMAL',
        }

    # ── Layer 2: Sub-group breakdown ──────────────────────────
    melspec_sub = {}
    for name, (s, e) in _MELSPEC_SUB.items():
        grp_abs = float(abs_shap[:, s:e].sum())
        mean_sv = float(shap_depresi[:, s:e].mean())
        melspec_sub[name] = {
            'contribution_pct': round(grp_abs / total * 100, 2) if total > 0 else 0.0,
            'direction':        'DEPRESI' if mean_sv >= 0 else 'NORMAL',
        }

    mfcc_sub = {}
    for name, (s, e) in _MFCC_SUB.items():
        grp_abs = float(abs_shap[:, s:e].sum())
        mean_sv = float(shap_depresi[:, s:e].mean())
        mfcc_sub[name] = {
            'contribution_pct': round(grp_abs / total * 100, 2) if total > 0 else 0.0,
            'direction':        'DEPRESI' if mean_sv >= 0 else 'NORMAL',
        }

    # Wav2Vec top-10 dims
    w2v_start    = 248
    w2v_abs_mean = abs_shap[:, w2v_start:].mean(axis=0)  # (768,)
    top10_idx    = np.argsort(w2v_abs_mean)[::-1][:10]
    wav2vec_top10 = [
        {
            'local_dim':     int(i),
            'global_dim':    int(i + w2v_start),
            'mean_abs_shap': round(float(w2v_abs_mean[i]), 6),
            'direction':     'DEPRESI' if float(shap_depresi[:, i+w2v_start].mean()) >= 0 else 'NORMAL',
        }
        for i in top10_idx
    ]

    layer2 = {
        'MelSpec_subgroups': melspec_sub,
        'MFCC_subgroups':    mfcc_sub,
        'Wav2Vec_top10':     wav2vec_top10,
    }

    # ── Layer 3: Waterfall — ambil segmen paling confident ───
    probs_seg  = pipeline.predict_proba(X_raw)[:, 1]   # (N,)
    # Pilih segmen dengan prob paling tinggi (paling decisive)
    pivot_idx  = int(np.argmax(np.abs(probs_seg - 0.5)))
    pivot_shap = shap_depresi[pivot_idx]               # (1016,)
    pivot_X    = X_scaled[pivot_idx]                   # (1016,)

    abs_pivot  = np.abs(pivot_shap)
    top_idx    = np.argsort(abs_pivot)[::-1][:top_n_waterfall]

    def _feat_label(idx):
        if idx < 128:  return ('MelSpec', f'Mel Band {idx+1}')
        if idx < 168:  return ('MFCC',    f'MFCC C{idx-128+1}')
        if idx < 208:  return ('MFCC',    f'Delta C{idx-168+1}')
        if idx < 248:  return ('MFCC',    f'Delta2 C{idx-208+1}')
        return ('Wav2Vec', f'Wav2Vec h{idx-248}')

    layer3 = []
    for rank, idx in enumerate(top_idx):
        grp, sub = _feat_label(int(idx))
        sv = float(pivot_shap[idx])
        layer3.append({
            'rank':              rank + 1,
            'feature_group':     grp,
            'feature_sub':       sub,
            'feature_idx':       int(idx),
            'shap_value':        round(sv, 6),
            'feature_val_scaled': round(float(pivot_X[idx]), 4),
            'direction':         'DEPRESI' if sv >= 0 else 'NORMAL',
            'magnitude':         round(abs(sv), 6),
        })

    # ── Layer 4: Plain language ───────────────────────────────
    dominant_grp = max(layer1, key=lambda k: layer1[k]['contribution_pct'])
    dominant_pct = layer1[dominant_grp]['contribution_pct']
    dominant_dir = layer1[dominant_grp]['direction']
    dominant_mfcc_sub = max(mfcc_sub, key=lambda k: mfcc_sub[k]['contribution_pct'])

    if dominant_grp == 'Wav2Vec':
        intro = (f"Pola embedding suara mendalam (Wav2Vec) menjadi faktor utama "
                 f"({dominant_pct:.1f}%) yang mendorong prediksi ke arah {dominant_dir}.")
    elif dominant_grp == 'MFCC':
        intro = (f"Karakteristik spektral suara (MFCC) menjadi faktor utama "
                 f"({dominant_pct:.1f}%) yang mendorong prediksi ke arah {dominant_dir}.")
    else:
        intro = (f"Pola energi frekuensi suara (Mel Spectrogram) menjadi faktor utama "
                 f"({dominant_pct:.1f}%) yang mendorong prediksi ke arah {dominant_dir}.")

    if 'Delta2' in dominant_mfcc_sub:
        mfcc_note = " Akselerasi perubahan bicara (MFCC Delta²) berperan, menunjukkan pola stabilitas bicara yang tidak tipikal."
    elif 'Delta' in dominant_mfcc_sub:
        mfcc_note = " Laju perubahan bicara (MFCC Delta) juga signifikan, mengindikasikan pola ritme bicara yang berbeda."
    else:
        mfcc_note = " Koefisien cepstral dasar mencerminkan karakteristik timbre dan kualitas vokal."

    layer4 = intro + mfcc_note

    return {
        'dominant_feature_group': dominant_grp,
        'layer1_group':           layer1,
        'layer2_subgroup':        layer2,
        'layer3_waterfall':       layer3,
        'layer4_text':            layer4,
        'baseline_prob_depresi':  round(expected_val, 4),
    }


# ============================================================
# 7. CONTOH INTEGRASI FLASK REST API
# ============================================================

def create_flask_app():
    """
    Contoh integrasi Flask.
    Jalankan: python template_website.py --serve

    Endpoint:
        POST /predict
            Form-data: audio=<file.wav>, participant_id=<str> (opsional)
            Response : JSON

        GET  /health
            Response : {"status": "ok", "model": "ML_Fusion_RF", ...}
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("[ERROR] Flask tidak terinstall. pip install flask")
        return None

    app = Flask(__name__)

    # Load SEKALI saat startup — jangan load per request!
    print("[STARTUP] Loading model, Wav2Vec & SHAP explainer...")
    _model_data   = load_model(MODEL_PATH)
    _w2v_model, _w2v_proc = load_wav2vec_model(device=DEVICE)
    _shap_explainer = load_shap_explainer(_model_data['pipeline'])
    print("[STARTUP] Ready!\n")

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            "status":    "ok",
            "model":     "ML_Fusion_RF",
            "scenario":  "Fusion (MelSpec + MFCC + Wav2Vec)",
            "fusion_dim": FUSION_DIM,
            "test_f1":   _model_data.get('test_f1', 'N/A'),
            "threshold": _model_data.get('threshold', 0.5),
            "device":    DEVICE,
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'audio' not in request.files:
            return jsonify({"error": "Field 'audio' tidak ditemukan di request"}), 400

        audio_file     = request.files['audio']
        participant_id = request.form.get('participant_id', 'unknown')

        # Simpan ke temp file
        import tempfile
        suffix = Path(audio_file.filename).suffix.lower() or '.wav'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = predict_depression(
                audio_path=tmp_path,
                model_path=MODEL_PATH,
                participant_id=participant_id,
                w2v_model=_w2v_model,
                w2v_processor=_w2v_proc,
                enable_xai=True,
                shap_explainer=_shap_explainer,
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(tmp_path)  # Hapus temp file

    return app


# ============================================================
# 7. CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depression Detection Inference — ML_Fusion_RF"
    )
    parser.add_argument(
        '--audio', type=str, default=None,
        help='Path ke file audio (.wav / .mp3)'
    )
    parser.add_argument(
        '--participant_id', type=str, default='unknown',
        help='ID partisipan (opsional)'
    )
    parser.add_argument(
        '--model', type=str, default=str(MODEL_PATH),
        help='Path ke best_model.pkl'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Simpan hasil JSON ke file ini (opsional)'
    )
    parser.add_argument(
        '--serve', action='store_true',
        help='Jalankan sebagai Flask REST API server di port 5000'
    )

    args = parser.parse_args()

    # ── Mode: Flask server ────────────────────────────────────
    if args.serve:
        flask_app = create_flask_app()
        if flask_app:
            print("[SERVER] Flask API berjalan di http://localhost:5000")
            print("[SERVER] Endpoint: POST /predict (form-data: audio=<file>)")
            print("[SERVER] Health  : GET  /health\n")
            flask_app.run(host='0.0.0.0', port=5000, debug=False)
        sys.exit(0)

    # ── Mode: Inferensi single audio ─────────────────────────
    if not args.audio:
        parser.print_help()
        print("\nContoh:")
        print("  python template_website.py --audio rekaman.wav")
        print("  python template_website.py --audio rekaman.mp3 --output hasil.json")
        print("  python template_website.py --serve")
        sys.exit(1)

    result = predict_depression(
        audio_path=args.audio,
        model_path=Path(args.model),
        participant_id=args.participant_id,
    )

    # Print JSON ke stdout
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Simpan ke file jika diminta
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[OUTPUT] Hasil disimpan ke: {args.output}")

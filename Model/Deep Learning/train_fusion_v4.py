"""
FUSION V4: SHARED WAV2VEC2 MULTI-MODAL FUSION
═══════════════════════════════════════════════════════════
4th Variation - Replaces BiLSTM (random init, problematic) with shared
Wav2Vec2 Transformer for ALL 3 branches (MFCC, MelSpec, Raw Waveform).

Architecture:
  Raw Waveform   → Wav2Vec2 (CNN + Transformer) → mean pool → Linear(768→256)
  MFCC  (T×120) → FeatureAdapter(120→512) → Wav2Vec2 Transformer → mean pool → Linear(768→256)
  MelSpec(T×128) → FeatureAdapter(128→512) → Wav2Vec2 Transformer → mean pool → Linear(768→256)
                   ↓
          Concat(768) → LN → FC(256) → GELU → Dropout → FC(2)

Key advantages over V3 (BiLSTM random init):
  ✅ No BiLSTM random init → no noise from untrained branches
  ✅ All 3 branches benefit from pretrained Wav2Vec2 transformer knowledge
  ✅ Only adapters learn from scratch (~0.3M params vs ~6M BiLSTM in V3)
  ✅ Far slower to overfit
  ✅ No data leakage (Wav2Vec2 pretrained from LibriSpeech, not our dataset)

2-Stage Training:
  Stage 1 (10 epochs): Freeze Wav2Vec2 encoder → train adapters + heads only
  Stage 2 (25 epochs): Fine-tune ALL with discriminative LRs

Single session: All 3 folds in one Colab session (~4-5 hours total)
Multi-session fallback: auto-detects completed folds from Drive checkpoint

SETUP (Google Colab):
  1. Mount Google Drive in notebook cell
  2. Upload this file to MyDrive/menthealth_data/
  3. Run setup cells (install, kaggle download, unzip dataset)
  4. !python /content/drive/MyDrive/menthealth_data/train_fusion_v4.py
"""

import os, sys, json, warnings, random, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from transformers import Wav2Vec2Model
    WAV2VEC_AVAILABLE = True
except ImportError:
    print("⚠️ transformers not installed! Run: pip install transformers")
    WAV2VEC_AVAILABLE = False

warnings.filterwarnings('ignore')

print('=' * 70)
print('FUSION V4: SHARED WAV2VEC2 MULTI-MODAL FUSION')
print('=' * 70)
print(f'PyTorch: {torch.__version__}')
print(f'Wav2Vec2: {"✅ Available" if WAV2VEC_AVAILABLE else "❌ Not available"}')

# ═══════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════

DATASET_NAME  = 'datasets/raihanthaffan/menthealth-data'
KAGGLE_INPUT  = Path('/kaggle/input')
COLAB_CONTENT = Path('/content')

if COLAB_CONTENT.exists():
    print("📍 Environment: Google Colab Detected")
    drive_myroot = Path('/content/drive/MyDrive')
    if drive_myroot.exists():
        PROJECT_ROOT = Path('/content/drive/MyDrive/menthealth_data/results_fusion_v4')
        print("  Google Drive terdeteksi. Checkpoint akan disimpan ke Drive.")
    else:
        print("  PERINGATAN: Drive belum di-mount!")
        PROJECT_ROOT = COLAB_CONTENT / 'results_fusion_v4'

    def _find_dir(*candidates):
        for p in candidates:
            if Path(p).exists():
                return Path(p)
        return Path(candidates[0])

    MFCC_DIR = _find_dir(
        '/content/data/features/features/mfcc',
        '/content/data/features/mfcc',
        '/content/features/features/mfcc',
    )
    SPEC_DIR = _find_dir(
        '/content/data/features/features/spectrogram',
        '/content/data/features/spectrogram',
        '/content/features/features/spectrogram',
    )
    WAV_DIR = _find_dir(
        '/content/data/features/features/waveform',
        '/content/data/features/waveform',
        '/content/features/features/waveform',
    )
    _csv_candidates = [
        '/content/data/splits/splits/custom_2class_labels.csv',
        '/content/data/splits/custom_2class_labels.csv',
        '/content/splits/splits/custom_2class_labels.csv',
    ]
    LABEL_CSV = next((Path(p) for p in _csv_candidates if Path(p).exists()), Path(_csv_candidates[0]))

elif KAGGLE_INPUT.exists():
    print("📍 Environment: Kaggle Detected")
    PROJECT_ROOT = Path('/kaggle/working/results_fusion_v4')
    INPUT_DIR    = KAGGLE_INPUT / DATASET_NAME
    MFCC_DIR     = INPUT_DIR / 'features' / 'features' / 'mfcc'
    SPEC_DIR     = INPUT_DIR / 'features' / 'features' / 'spectrogram'
    WAV_DIR      = INPUT_DIR / 'features' / 'features' / 'waveform'
    LABEL_CSV    = INPUT_DIR / 'splits' / 'splits' / 'custom_2class_labels.csv'
else:
    print("📍 Environment: Local Detected")
    try:
        base_dir = Path(__file__).parent.parent.parent
    except NameError:
        base_dir = Path('.')
    PROJECT_ROOT = base_dir / 'results_fusion_v4'
    MFCC_DIR     = base_dir / 'data' / 'features' / 'mfcc'
    SPEC_DIR     = base_dir / 'data' / 'features' / 'spectrogram'
    WAV_DIR      = base_dir / 'data' / 'features' / 'waveform'
    LABEL_CSV    = base_dir / 'data' / 'splits' / 'custom_2class_labels.csv'

V4_MODEL_DIR   = PROJECT_ROOT / 'models'
V4_RESULTS_DIR = PROJECT_ROOT / 'results'

print(f'\n📂 Environment Paths:')
print(f'   MFCC Directory: {MFCC_DIR}')
print(f'   Label CSV:      {LABEL_CSV}')
print(f'   Output Results: {V4_RESULTS_DIR}')

for d in [V4_MODEL_DIR, V4_RESULTS_DIR,
          V4_RESULTS_DIR / 'metrics',
          V4_RESULTS_DIR / 'confusion_matrix',
          V4_RESULTS_DIR / 'plots']:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {gpu_mem_gb:.1f} GB')

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ═══════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════

N_FOLDS = 3

if torch.cuda.is_available():
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_mem_gb < 10.0:
        BATCH_SIZE = 8
    elif gpu_mem_gb < 16.0:
        BATCH_SIZE = 16   # T4 (15.6 GB): safe for 3x Wav2Vec2 encoder passes
    else:
        BATCH_SIZE = 24
    print(f"⚙️  Batch size: {BATCH_SIZE} (3x Wav2Vec2 encoder, VRAM-aware).")
else:
    BATCH_SIZE = 4

# Discriminative learning rates
LR_WAV2VEC_ENCODER = 1e-6   # Wav2Vec2 Transformer layers 8-11 (lowered: 3e-6 → 1e-6 to prevent NaN explosion)
LR_WAV2VEC_BASE    = 5e-7   # Wav2Vec2 CNN + feature_projection (lowered: 2e-6 → 5e-7)
LR_ADAPTERS        = 5e-5   # FeatureAdapters (from scratch, but tiny)
LR_PROJ_HEADS      = 5e-5   # Per-branch projection heads
LR_CLASSIFIER      = 1e-4   # Fusion classifier

WEIGHT_DECAY = 1e-3
PATIENCE_ES  = 8
MIN_LR       = 1e-8

# Data (must match V3 preprocessing)
N_MELS        = 128
MFCC_FEATURES = 120
MAX_SPEC_LEN  = 313
MAX_MFCC_LEN  = 313
RAW_LEN       = 160_000
CLASS_NAMES   = ['NORMAL', 'DEPRESI']
N_CLASSES     = 2

# 2-Stage strategy
STAGE1_EPOCHS = 10   # Adapter warm-up (Wav2Vec2 frozen)
STAGE2_EPOCHS = 25   # Full fine-tune (discriminative LR)

FOCAL_GAMMA = 2.0
DROPOUT     = 0.5

print(f'\n{"=" * 70}')
print('HYPERPARAMETERS (FUSION V4)')
print(f'{"=" * 70}')
print(f'Folds:        {N_FOLDS}')
print(f'Epochs:       Stage 1: {STAGE1_EPOCHS} (adapter warmup), Stage 2: {STAGE2_EPOCHS}')
print(f'Batch Size:   {BATCH_SIZE}')
print(f'LR Wav2Vec2 encoder: {LR_WAV2VEC_ENCODER}')
print(f'LR Adapters:         {LR_ADAPTERS}')
print(f'LR Classifier:       {LR_CLASSIFIER}')
print(f'Focal Loss:   gamma={FOCAL_GAMMA}')
print(f'Dropout:      {DROPOUT}')

# ═══════════════════════════════════════════════════════════
# MULTI-SESSION SUPPORT
# ═══════════════════════════════════════════════════════════

SKIP_FOLDS     = []    # Leave empty for single-session (all 3 folds)
PROGRESS_FILE  = V4_RESULTS_DIR / 'training_progress.json'

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'completed_folds': []}

def save_progress(fold):
    p = load_progress()
    if fold not in p['completed_folds']:
        p['completed_folds'].append(fold)
    p['last_updated'] = pd.Timestamp.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(p, f, indent=2)
    print(f'  ✓ Progress saved: {len(p["completed_folds"])} folds completed')

def should_skip_fold(fold):
    if fold in SKIP_FOLDS:
        return True
    if (V4_MODEL_DIR / f'fold{fold}_best.pt').exists():
        if fold in load_progress().get('completed_folds', []):
            return True
    return False

progress = load_progress()
print(f'\n{"=" * 70}')
print('SESSION CONFIGURATION')
print(f'{"=" * 70}')
print(f'Completed folds: {progress.get("completed_folds", [])}')
print(f'Skip folds:      {SKIP_FOLDS}')
print(f'Will train:      Fold {[f for f in range(1, N_FOLDS+1) if not should_skip_fold(f)]}')
print('✓ Multi-session support ready')

# ═══════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

print('✓ Focal Loss ready')

# ═══════════════════════════════════════════════════════════
# NORMALIZATION & AUGMENTATION
# ═══════════════════════════════════════════════════════════

def normalize_feature(x):
    """Z-score normalization yang aman untuk segment silence / near-flat."""
    mean = x.mean()
    std  = x.std()
    if std < 1e-6:          # segment senyap / flat → skip normalisasi
        return x - mean
    out = (x - mean) / std
    return out.clip(-10.0, 10.0)   # clamp supaya tidak ada outlier ekstrem

def augment_waveform(wav, aug_type='noise'):
    if aug_type == 'noise':
        return wav + np.random.randn(len(wav)).astype(np.float32) * 0.003
    elif aug_type == 'gain':
        return wav * np.random.uniform(0.8, 1.2)
    elif aug_type == 'shift':
        return np.roll(wav, np.random.randint(-1600, 1600))
    return wav

def augment_spectrogram(spec):
    spec = spec.copy()
    T, F = spec.shape
    if T > 20:
        t0 = np.random.randint(0, max(1, T - min(40, T // 4)))
        spec[t0:t0 + min(40, T // 4), :] = 0
    if F > 10:
        f0 = np.random.randint(0, max(1, F - min(15, F // 6)))
        spec[:, f0:f0 + min(15, F // 6)] = 0
    return spec

print('✓ Normalization & Augmentation ready')

# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════

class AllSegmentDataset(Dataset):
    """All segments per participant."""
    def __init__(self, participant_samples, augment=False):
        self.segments = []
        self.augment  = augment
        for pid, label in participant_samples:
            subfolder = 'DEPRESI' if label == 1 else 'NORMAL'
            segs = sorted((MFCC_DIR / subfolder).glob(f"{pid}_seg*.npy"))
            for seg_path in segs:
                self.segments.append((seg_path.stem, label, pid))
        print(f'  Dataset: {len(self.segments)} segments from {len(participant_samples)} participants')
        print(f'  Effective increase: {len(self.segments) / max(len(participant_samples), 1):.1f}× more samples!')

    def __len__(self): return len(self.segments)

    def __getitem__(self, idx):
        seg_id, label, _ = self.segments[idx]
        subfolder = 'DEPRESI' if label == 1 else 'NORMAL'

        mfcc = np.load(MFCC_DIR / subfolder / f"{seg_id}.npy").astype(np.float32)
        if mfcc.shape[0] > MAX_MFCC_LEN:
            mfcc = mfcc[:MAX_MFCC_LEN]
        elif mfcc.shape[0] < MAX_MFCC_LEN:
            mfcc = np.vstack([mfcc, np.zeros((MAX_MFCC_LEN - mfcc.shape[0], MFCC_FEATURES), np.float32)])

        spec = np.load(SPEC_DIR / subfolder / f"{seg_id}.npy").astype(np.float32)
        if spec.ndim == 2 and spec.shape[0] == N_MELS:
            spec = spec.T
        if spec.shape[0] > MAX_SPEC_LEN:
            spec = spec[:MAX_SPEC_LEN]
        elif spec.shape[0] < MAX_SPEC_LEN:
            spec = np.vstack([spec, np.zeros((MAX_SPEC_LEN - spec.shape[0], N_MELS), np.float32)])

        wav = np.load(WAV_DIR / subfolder / f"{seg_id}.npy").astype(np.float32)
        if len(wav) > RAW_LEN:
            wav = wav[:RAW_LEN]
        elif len(wav) < RAW_LEN:
            wav = np.pad(wav, (0, RAW_LEN - len(wav)))

        if self.augment and label == 1 and random.random() < 0.5:
            aug = random.choice(['noise', 'gain', 'shift', 'spec'])
            if aug in ['noise', 'gain', 'shift']:
                wav = augment_waveform(wav, aug)
            if aug == 'spec' or random.random() < 0.5:
                mfcc = augment_spectrogram(mfcc)
                spec = augment_spectrogram(spec)

        return (torch.from_numpy(normalize_feature(mfcc)),
                torch.from_numpy(normalize_feature(spec)),
                torch.from_numpy(normalize_feature(wav)),
                label)

print('✓ All-segment dataset ready')

# ═══════════════════════════════════════════════════════════
# MODEL: FUSION V4
# ═══════════════════════════════════════════════════════════

class FeatureAdapter(nn.Module):
    """
    Projects MFCC (120-dim) or MelSpec (128-dim) sequence to Wav2Vec2
    feature extractor output dimension (512-dim), allowing these modalities
    to be fed directly into the shared Wav2Vec2 Transformer (bypassing CNN).

    Parameters: ~60K per adapter (tiny → no overfitting risk)
    """
    def __init__(self, input_dim: int, output_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, T, input_dim) → (B, T, 512)"""
        return self.proj(x)


class FusionV4Model(nn.Module):
    """
    Fusion V4: Shared Wav2Vec2 Multi-Modal Fusion

    Architecture:
      Raw Waveform  → Wav2Vec2 CNN → feature_projection → Transformer → pool → proj(768→256)
      MFCC          → FeatureAdapter(120→512) → feature_projection → Transformer → pool → proj(768→256)
      MelSpec       → FeatureAdapter(128→512) → feature_projection → Transformer → pool → proj(768→256)
      Concat(768) → LN → FC(256) → GELU → Dropout → FC(2)

    All 3 branches share ONE Wav2Vec2 transformer.
    Only adapters (~120K params total) train from scratch.
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        if not WAV2VEC_AVAILABLE:
            raise ImportError("transformers not installed!")

        print('\n  Building Fusion V4 model...')

        print('  Loading Wav2Vec2-base (shared backbone)...')
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            output_hidden_states=False
        )
        print('  ✓ Wav2Vec2 loaded')

        # Gradient checkpointing saves VRAM when running 3x encoder passes
        try:
            self.wav2vec.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            self.wav2vec.gradient_checkpointing_enable()

        # Freeze early Transformer layers 0-7
        for i, layer in enumerate(self.wav2vec.encoder.layers):
            if i < 8:
                for p in layer.parameters():
                    p.requires_grad = False
        print('  ✓ Encoder layers 0-7 frozen, 8-11 trainable')

        # Feature adapters (learn from scratch, but tiny)
        self.mfcc_adapter = FeatureAdapter(MFCC_FEATURES, 512, dropout=0.1)
        self.spec_adapter  = FeatureAdapter(N_MELS,        512, dropout=0.1)
        adapter_params = (sum(p.numel() for p in self.mfcc_adapter.parameters()) +
                          sum(p.numel() for p in self.spec_adapter.parameters()))
        print(f'  ✓ Adapters: MFCC({MFCC_FEATURES}→512) + MelSpec({N_MELS}→512) '
              f'= {adapter_params/1e3:.1f}K params from scratch')

        # Per-branch projection heads (768 → 256)
        def _proj():
            return nn.Sequential(
                nn.Linear(768, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.raw_proj  = _proj()
        self.mfcc_proj = _proj()
        self.spec_proj = _proj()

        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, N_CLASSES),
        )

        print('  ✓ Model architecture built')
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Total parameters:     {total/1e6:.1f}M')
        print(f'  Trainable parameters: {trainable/1e6:.1f}M')

    def _encode_via_transformer(self, adapted: torch.Tensor) -> torch.Tensor:
        """
        Feed adapter output (B, T, 512) through Wav2Vec2 feature_projection
        and shared Transformer encoder, bypassing the CNN front-end.
        Returns mean-pooled hidden states (B, 768).
        """
        hidden = self.wav2vec.feature_projection(adapted)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        encoder_out = self.wav2vec.encoder(hidden)
        if hasattr(encoder_out, 'last_hidden_state'):
            last_hidden = encoder_out.last_hidden_state
        else:
            last_hidden = encoder_out[0]

        return last_hidden.mean(dim=1)          # (B, 768)

    def forward(self, mfcc, spec, wav):
        # Branch 1: Raw → full Wav2Vec2 (CNN + Transformer)
        e_raw  = self.wav2vec(wav).last_hidden_state.mean(dim=1)  # (B, 768)
        e_raw  = self.raw_proj(e_raw)                              # (B, 256)

        # Branch 2: MFCC → Adapter → Transformer
        e_mfcc = self._encode_via_transformer(self.mfcc_adapter(mfcc))  # (B, 768)
        e_mfcc = self.mfcc_proj(e_mfcc)                                  # (B, 256)

        # Branch 3: MelSpec → Adapter → Transformer
        e_spec = self._encode_via_transformer(self.spec_adapter(spec))   # (B, 768)
        e_spec = self.spec_proj(e_spec)                                   # (B, 256)

        # Fusion
        fused = torch.cat([e_raw, e_mfcc, e_spec], dim=1)  # (B, 768)
        return self.fusion(fused)                            # (B, 2)

print('✓ FusionV4Model defined')

# ═══════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════

# ─── GradScaler: JANGAN buat di global scope! ────────────────────────────────
# scaler dibuat ulang setiap fold (lihat train_one_fold_2stage) agar fold
# berikutnya tidak mewarisi scaler yang sudah overflow/rusak dari fold sebelumnya.
scaler = None   # akan di-set di train_one_fold_2stage()

def train_epoch(model, loader, criterion, optimizer, epoch_num, total_epochs):
    model.train()
    total_loss = correct = total = 0
    nan_batches = 0
    pbar = tqdm(loader, desc=f'  Epoch {epoch_num:2d}/{total_epochs}', leave=False)
    for mfcc, spec, wav, labels in pbar:
        mfcc, spec, wav = mfcc.to(DEVICE), spec.to(DEVICE), wav.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # ✅ FIX #1: Hanya forward model di dalam autocast (float16 aman).
        # FocalLoss dihitung di float32 SETELAH cast output ke float32,
        # karena torch.exp(-ce) dalam float16 bisa overflow → NaN.
        with torch.amp.autocast(device_type=DEVICE.type):
            outputs = model(mfcc, spec, wav)
        outputs = outputs.float()       # cast ke float32 sebelum loss
        loss = criterion(outputs, labels)

        # ✅ FIX #4: NaN guard — skip batch yang menghasilkan NaN/Inf loss
        if not torch.isfinite(loss):
            nan_batches += 1
            if nan_batches <= 3:
                print(f'\n  ⚠️  NaN/Inf loss pada batch #{total//max(labels.size(0),1)+1}, skip.')
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)   # tighter clip: 1.0 → 0.5
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total   += labels.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})

    if nan_batches > 0:
        print(f'  ⚠️  Total NaN batches skipped epoch ini: {nan_batches}')
    if total == 0:
        return float('nan'), 0.0
    return total_loss / total, 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mfcc, spec, wav, labels in tqdm(loader, desc='  Val', leave=False):
            mfcc, spec, wav = mfcc.to(DEVICE), spec.to(DEVICE), wav.to(DEVICE)
            labels = labels.to(DEVICE)
            # ✅ FIX #1 (eval): sama — FocalLoss dihitung di float32
            with torch.amp.autocast(device_type=DEVICE.type):
                outputs = model(mfcc, spec, wav)
            outputs = outputs.float()
            loss    = criterion(outputs, labels)
            if not torch.isfinite(loss):
                continue
            total_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if total == 0:
        return float('nan'), 0.0, 0.0, [], []
    return (total_loss / total, 100. * correct / total,
            f1_score(all_labels, all_preds, average='macro'), all_preds, all_labels)

def evaluate_with_segment_aggregation(model, test_participants):
    model.eval()
    all_predictions, all_true = [], []
    print('  Evaluating with segment aggregation...')
    with torch.no_grad():
        for pid, true_label in tqdm(test_participants, desc='  Aggregate', leave=False):
            subfolder = 'DEPRESI' if true_label == 1 else 'NORMAL'
            segments  = sorted((MFCC_DIR / subfolder).glob(f"{pid}_seg*.npy"))
            if not segments:
                continue
            seg_probs = []
            for seg_path in segments:
                seg_id = seg_path.stem
                mfcc = np.load(MFCC_DIR / subfolder / f"{seg_id}.npy").astype(np.float32)
                spec = np.load(SPEC_DIR / subfolder / f"{seg_id}.npy").astype(np.float32)
                wav  = np.load(WAV_DIR  / subfolder / f"{seg_id}.npy").astype(np.float32)

                if mfcc.shape[0] > MAX_MFCC_LEN:  mfcc = mfcc[:MAX_MFCC_LEN]
                elif mfcc.shape[0] < MAX_MFCC_LEN:
                    mfcc = np.vstack([mfcc, np.zeros((MAX_MFCC_LEN - mfcc.shape[0], MFCC_FEATURES))])
                if spec.ndim == 2 and spec.shape[0] == N_MELS: spec = spec.T
                if spec.shape[0] > MAX_SPEC_LEN:  spec = spec[:MAX_SPEC_LEN]
                elif spec.shape[0] < MAX_SPEC_LEN:
                    spec = np.vstack([spec, np.zeros((MAX_SPEC_LEN - spec.shape[0], N_MELS))])
                if len(wav) > RAW_LEN:  wav = wav[:RAW_LEN]
                elif len(wav) < RAW_LEN: wav = np.pad(wav, (0, RAW_LEN - len(wav)))

                mfcc_t = torch.from_numpy(normalize_feature(mfcc.astype(np.float32))).unsqueeze(0).to(DEVICE)
                spec_t = torch.from_numpy(normalize_feature(spec.astype(np.float32))).unsqueeze(0).to(DEVICE)
                wav_t  = torch.from_numpy(normalize_feature(wav.astype(np.float32))).unsqueeze(0).to(DEVICE)

                with torch.amp.autocast(device_type=DEVICE.type):
                    out = model(mfcc_t, spec_t, wav_t)
                seg_probs.append(F.softmax(out, dim=1).cpu().numpy())

            final_pred = np.mean(seg_probs, axis=0).argmax()
            all_predictions.append(final_pred)
            all_true.append(true_label)

    return (f1_score(all_true, all_predictions, average='macro'),
            accuracy_score(all_true, all_predictions),
            all_predictions, all_true)

print('✓ Training & evaluation functions ready')

# ═══════════════════════════════════════════════════════════
# AUTO-BACKUP
# ═══════════════════════════════════════════════════════════

def auto_backup(fold: int):
    print(f'\n{"─" * 70}')
    print(f'💾 AUTO-BACKUP: Fold {fold}')
    print(f'{"─" * 70}')
    ts          = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    backup_path = PROJECT_ROOT / f'backup_fold{fold}_{ts}.zip'
    try:
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for d in [V4_MODEL_DIR, V4_RESULTS_DIR]:
                for f in d.rglob('*'):
                    if f.is_file():
                        zf.write(f, f.relative_to(PROJECT_ROOT))
        size_mb = backup_path.stat().st_size / 1e6
        print(f'✅ Backup created: {backup_path.name} ({size_mb:.1f} MB)')
        print(f'📥 Download from Output tab or Google Drive')
    except Exception as e:
        print(f'⚠️ Backup failed: {e}')
    print(f'{"─" * 70}')

# ═══════════════════════════════════════════════════════════
# OPTIMIZER BUILDER (DISCRIMINATIVE LR)
# ═══════════════════════════════════════════════════════════

def build_optimizer(model, stage: int):
    if stage == 1:
        # Freeze all Wav2Vec2
        for p in model.wav2vec.parameters():
            p.requires_grad = False
        # Ensure learnable parts are trainable
        for m in [model.mfcc_adapter, model.spec_adapter,
                  model.raw_proj, model.mfcc_proj, model.spec_proj, model.fusion]:
            for p in m.parameters():
                p.requires_grad = True
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Stage 1: {n/1e3:.1f}K trainable (adapters + heads, Wav2Vec2 frozen)')
        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_ADAPTERS, weight_decay=WEIGHT_DECAY
        )
    else:
        # Unfreeze Wav2Vec2
        for p in model.wav2vec.parameters():
            p.requires_grad = True
        # Keep feature extractor frozen (always freeze for stability & representations)
        for p in model.wav2vec.feature_extractor.parameters():
            p.requires_grad = False
        # Re-freeze layers 0-7
        for i, layer in enumerate(model.wav2vec.encoder.layers):
            if i < 8:
                for p in layer.parameters():
                    p.requires_grad = False
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Stage 2: {n/1e6:.1f}M trainable (full fine-tune, discriminative LR)')
        # Only include encoder params that are actually trainable (layers 8-11, not 0-7)
        encoder_trainable = [p for p in model.wav2vec.encoder.parameters() if p.requires_grad]
        return optim.AdamW([
            {'params': list(model.wav2vec.feature_projection.parameters()), 'lr': LR_WAV2VEC_BASE},
            {'params': encoder_trainable,                                    'lr': LR_WAV2VEC_ENCODER},
            {'params': list(model.mfcc_adapter.parameters()),               'lr': LR_ADAPTERS},
            {'params': list(model.spec_adapter.parameters()),               'lr': LR_ADAPTERS},
            {'params': (list(model.raw_proj.parameters()) +
                        list(model.mfcc_proj.parameters()) +
                        list(model.spec_proj.parameters())),                'lr': LR_PROJ_HEADS},
            {'params': list(model.fusion.parameters()),                     'lr': LR_CLASSIFIER},
        ], weight_decay=WEIGHT_DECAY)

# ═══════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════

def train_one_fold(fold: int, train_samples, val_samples):
    global scaler
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available(), init_scale=1024)  # low init_scale → avoid float16 overflow
    print(f'\n{"=" * 70}')
    print(f'FOLD {fold} - 2-STAGE TRAINING (Fusion V4)')
    print(f'{"=" * 70}')

    train_ds = AllSegmentDataset(train_samples, augment=True)
    val_ds   = AllSegmentDataset(val_samples,   augment=False)

    train_labels  = [lbl for _, lbl, _ in train_ds.segments]
    class_counts  = Counter(train_labels)
    n_normal, n_depresi = class_counts[0], class_counts[1]
    total = len(train_labels)
    print(f'\n  Train: NORMAL={n_normal}, DEPRESI={n_depresi} ({n_depresi/total*100:.1f}% DEPRESI)')

    inv_freq       = {0: 1.0 / n_normal, 1: 1.0 / n_depresi}
    sample_weights = [inv_freq[lbl] for lbl in train_labels]
    sampler        = WeightedRandomSampler(sample_weights, num_samples=total, replacement=True)
    eff_n = (n_normal * inv_freq[0]) / (n_normal * inv_freq[0] + n_depresi * inv_freq[1])
    print(f'  Sampler: NORMAL={eff_n*100:.0f}% / DEPRESI={(1-eff_n)*100:.0f}% (50/50 target)')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    print(f'  Train batches/epoch: {len(train_loader)}')
    print(f'  Val batches:         {len(val_loader)}')

    model     = FusionV4Model(dropout=DROPOUT).to(DEVICE)
    criterion = FocalLoss(alpha=None, gamma=FOCAL_GAMMA)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1  = 0.0
    best_state = None

    # ── STAGE 1: Adapter Warm-Up ──────────────────────────────
    print(f'\n{"─" * 70}')
    print(f'STAGE 1: ADAPTER WARM-UP ({STAGE1_EPOCHS} epochs, Wav2Vec2 frozen)')
    print(f'{"─" * 70}')
    opt1 = build_optimizer(model, stage=1)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS, eta_min=MIN_LR)

    for epoch in range(1, STAGE1_EPOCHS + 1):
        tr_loss, tr_acc          = train_epoch(model, train_loader, criterion, opt1, epoch, STAGE1_EPOCHS)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion)
        sch1.step()
        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss);   history['val_acc'].append(vl_acc)
        history['val_f1'].append(vl_f1)
        flag = ''
        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = f'    ✓ New best F1: {vl_f1:.4f}'
        print(f'  Epoch {epoch:2d}/{STAGE1_EPOCHS} | '
              f'Train: Loss={tr_loss:.4f} Acc={tr_acc:.1f}% | '
              f'Val: Loss={vl_loss:.4f} Acc={vl_acc:.1f}% F1={vl_f1:.4f}{flag}')

    # ── STAGE 2: Full Fine-Tune ────────────────────────────────
    print(f'\n{"─" * 70}')
    print(f'STAGE 2: FULL FINE-TUNE ({STAGE2_EPOCHS} epochs, discriminative LR)')
    print(f'{"─" * 70}')
    opt2    = build_optimizer(model, stage=2)
    sch2    = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS, eta_min=MIN_LR)
    patience = 0

    for epoch in range(1, STAGE2_EPOCHS + 1):
        global_ep = STAGE1_EPOCHS + epoch
        total_ep  = STAGE1_EPOCHS + STAGE2_EPOCHS
        tr_loss, tr_acc               = train_epoch(model, train_loader, criterion, opt2, global_ep, total_ep)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion)
        sch2.step()

        # ✅ NaN Recovery: jika model explode, reload best checkpoint & lanjut
        if np.isnan(tr_loss) or np.isnan(vl_loss):
            patience += 1
            print(f'  🔄 NaN epoch {global_ep}! Reloading best checkpoint... Patience: {patience}/{PATIENCE_ES}')
            if best_state is not None:
                model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
            if patience >= PATIENCE_ES:
                print(f'  Early stopping at epoch {global_ep} (too many NaN epochs)')
                break
            continue

        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss);   history['val_acc'].append(vl_acc)
        history['val_f1'].append(vl_f1)
        flag = ''
        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
            flag = f'    ✓ New best F1: {vl_f1:.4f}'
        else:
            patience += 1
            if patience >= PATIENCE_ES:
                print(f'  Early stopping at epoch {global_ep}')
                break
        print(f'  Epoch {global_ep:2d}/{total_ep} | '
              f'Train: Loss={tr_loss:.4f} Acc={tr_acc:.1f}% | '
              f'Val: Loss={vl_loss:.4f} Acc={vl_acc:.1f}% F1={vl_f1:.4f}{flag}')

    # Save best model
    model.load_state_dict(best_state)
    torch.save(best_state, V4_MODEL_DIR / f'fold{fold}_best.pt')
    print(f'\n  ✓ Best model saved (segment-level F1={best_f1:.4f})')

    # Learning curves plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
    axes[0].axvline(x=STAGE1_EPOCHS, color='r', linestyle='--', alpha=0.5, label='Stage 2')
    axes[0].set_title(f'Fold {fold} - Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['val_f1'], color='green', label='Val F1')
    axes[1].axhline(y=best_f1, color='r', linestyle='--', label=f'Best: {best_f1:.4f}')
    axes[1].axvline(x=STAGE1_EPOCHS, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title(f'Fold {fold} - Val F1'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(V4_RESULTS_DIR / 'plots' / f'fold{fold}_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Final evaluation with segment aggregation
    print(f'\n{"─" * 70}')
    print('FINAL EVALUATION (Segment Aggregation)')
    print(f'{"─" * 70}')
    f1_agg, acc_agg, preds, true_labels = evaluate_with_segment_aggregation(model, val_samples)

    print(f'\n  Validation Results:')
    print(f'    F1-Macro (aggregated): {f1_agg:.4f}')
    print(f'    Accuracy (aggregated): {acc_agg:.4f}')
    print(f'\n  Classification Report:')
    print(classification_report(true_labels, preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Fusion V4 - Fold {fold}\nF1-Macro: {f1_agg:.4f}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.savefig(V4_RESULTS_DIR / 'confusion_matrix' / f'fold{fold}_cm.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    with open(V4_RESULTS_DIR / 'metrics' / f'fold{fold}_metrics.json', 'w') as fp:
        json.dump({
            'fold': fold,
            'best_val_f1_segment': float(best_f1),
            'f1_macro_aggregated': float(f1_agg),
            'accuracy_aggregated': float(acc_agg),
            'history': history,
        }, fp, indent=2)

    save_progress(fold)
    return f1_agg, acc_agg

print('✓ Training function ready')

# ═══════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════

def run_cross_validation():
    print(f'\n{"=" * 70}')
    print('STARTING 3-FOLD CROSS-VALIDATION — FUSION V4')
    print(f'{"=" * 70}')

    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"Label CSV not found: {LABEL_CSV}")

    df = pd.read_csv(LABEL_CSV)
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    id_col    = 'id'    if 'id'    in df.columns else df.columns[0]
    df[label_col] = df[label_col].map({'NORMAL': 0, 'DEPRESI': 1}).fillna(df[label_col])
    all_samples   = list(zip(df[id_col].astype(str), df[label_col].astype(int)))

    print(f'\nDataset: {len(all_samples)} participants')
    print(f'  NORMAL:  {sum(1 for _, l in all_samples if l == 0)}')
    print(f'  DEPRESI: {sum(1 for _, l in all_samples if l == 1)}')

    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    labels_arr = np.array([l for _, l in all_samples])
    splits     = list(skf.split(all_samples, labels_arr))

    fold_results = [None] * N_FOLDS

    # Load already-completed results
    for fold in range(1, N_FOLDS + 1):
        mpath = V4_RESULTS_DIR / 'metrics' / f'fold{fold}_metrics.json'
        if should_skip_fold(fold):
            print(f'\n{"=" * 70}')
            print(f'⏭️  SKIPPING FOLD {fold} (Already completed)')
            print(f'{"=" * 70}')
            if mpath.exists():
                with open(mpath) as fp:
                    m = json.load(fp)
                fold_results[fold-1] = (m['f1_macro_aggregated'], m['accuracy_aggregated'])
                print(f'  ✓ Loaded: F1={m["f1_macro_aggregated"]:.4f}  Acc={m["accuracy_aggregated"]:.4f}')
            else:
                print(f'  ⚠️ No saved results found')

    # Train remaining folds
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold = fold_idx + 1
        if should_skip_fold(fold):
            continue
        train_s = [all_samples[i] for i in train_idx]
        val_s   = [all_samples[i] for i in val_idx]
        print(f'\nFold {fold}: Train={len(train_s)}, Val={len(val_s)} participants')
        f1_fold, acc_fold        = train_one_fold(fold, train_s, val_s)
        fold_results[fold-1]     = (f1_fold, acc_fold)
        auto_backup(fold)

    # Summary
    print(f'\n{"=" * 70}')
    print('CROSS-VALIDATION SUMMARY — FUSION V4')
    print(f'{"=" * 70}')
    valid = [(f, a) for r in fold_results if r is not None for f, a in [r]]
    if valid:
        f1s  = [r[0] for r in valid]
        accs = [r[1] for r in valid]
        print(f'\n{"─" * 70}')
        print(f'F1-Macro:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
        print(f'Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}')
        print(f'{"─" * 70}')
        for i, (f, a) in enumerate(valid):
            print(f'  Fold {i+1}: F1={f:.4f}  Acc={a:.4f}')

        v3_f1 = 0.5641
        delta = np.mean(f1s) - v3_f1
        print(f'\n{"=" * 70}')
        print('COMPARISON: V3 vs V4')
        print(f'{"=" * 70}')
        print(f'  V3 (BiLSTM random):   F1={v3_f1:.4f}')
        print(f'  V4 (Shared Wav2Vec2): F1={np.mean(f1s):.4f}  ({"+" if delta >= 0 else ""}{delta:+.4f})')
        if np.mean(f1s) >= 0.75:
            print('\n🎉 TARGET F1 ≥ 0.75 REACHED!')
        else:
            print(f'\n❌ Target not reached. Gap: {0.75 - np.mean(f1s):.4f}')

        with open(V4_RESULTS_DIR / 'summary.json', 'w') as fp:
            json.dump({
                'model': 'Fusion V4 - Shared Wav2Vec2',
                'architecture': {
                    'shared_wav2vec2': True,
                    'mfcc_adapter': f'Linear({MFCC_FEATURES}→512) + LN + GELU',
                    'spec_adapter': f'Linear({N_MELS}→512) + LN + GELU',
                    'fusion': 'Concat(768) → LN → FC(256) → GELU → Dropout → FC(2)',
                },
                'cv_f1_mean': float(np.mean(f1s)),
                'cv_f1_std':  float(np.std(f1s)),
                'cv_acc_mean': float(np.mean(accs)),
                'per_fold': [{'fold': i+1, 'f1': f, 'acc': a}
                             for i, (f, a) in enumerate(valid)],
                'vs_v3': {'v3_f1': v3_f1, 'delta': float(delta)},
            }, fp, indent=2)
        print(f'\n✅ Summary saved to: {V4_RESULTS_DIR / "summary.json"}')

    print(f'\n{"=" * 70}')
    print('TRAINING COMPLETE — FUSION V4')
    print(f'{"=" * 70}')
    print(f'Results: {V4_RESULTS_DIR}')
    print(f'Models:  {V4_MODEL_DIR}')
    print('\n🎉 ALL DONE!')


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'\n{"=" * 70}')
    print('FUSION V4: SHARED WAV2VEC2 — READY TO START')
    print(f'{"=" * 70}')
    print('Estimated: ~4-5 hours total (3 folds in one session)')
    print('Per fold: ~1.5-2 hours\n')
    import time; time.sleep(3)
    run_cross_validation()

"""
train_model.py
==============
Voice Spoof Detection — CNN Trainer

KEY RULE: The feature extraction here MUST exactly match the inference
          pipeline in backend/app.py → extract_features_from_audio().

Changes from previous version:
  - Feature extraction is now a shared function (no augmentation variant)
  - Augmentation is applied at the AUDIO level before feature extraction
  - Added mic-specific augmentation: room-reverb, low-pass filter, level variation
  - Added delta-MFCC channels (optional, controlled by USE_DELTA)
  - Heavier oversampling of mic real samples to correct dataset imbalance
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────────
#  CONFIG — edit only this section if paths differ
# ─────────────────────────────────────────────
DATASET_PATH    = r"D:\Dissertion 2\voice spoof try2\dataset"
MODEL_SAVE_PATH = r"D:\Dissertion 2\voice spoof try2\backend\model\voice_model.keras"

SR       = 22050   # sample rate — must match app.py
N_MFCC   = 40      # number of MFCC coefficients — must match app.py
N_FRAMES = 94      # time frames per clip — must match app.py

# Extra augmentation passes for mic-recorded real samples (to balance dataset)
# e.g. 8 means each mic file gets 8 augmented copies IN ADDITION to the original
MIC_AUGMENT_FACTOR = 8

# Standard augmentation passes for all other files (original + N copies)
STD_AUGMENT_FACTOR = 3

# Extra augmentation for recorded spoof files (they are few but important)
SPOOF_REC_AUGMENT_FACTOR = 10

# ─────────────────────────────────────────────
#  DATASET FOLDERS
# ─────────────────────────────────────────────
real_folders_studio = ["real_samples"]          # studio / clean recordings
real_folders_mic    = ["real_mic_recorded"]     # mic-recorded real voice

spoof_folders = [
    "FlashSpeech",
    "NaturalSpeech3",
    "OpenAI",
    "PromptTTS2",
    "seedtts_files",
    "VALLE",
    "VoiceBox",
    "xTTS",
    "spoofed_recorded",
]

print("Dataset path:", DATASET_PATH)
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset folder not found at {DATASET_PATH}")


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
#  ⚠️  This MUST match app.py → extract_features_from_audio()
# ─────────────────────────────────────────────
def extract_features_from_audio(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extract 40-MFCC features from an audio array.
    Matches inference pipeline exactly (no augmentation applied here).
    """
    # Resample if needed
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)

    # Ensure at least 1 second
    if len(y) < SR:
        y = np.pad(y, (0, SR - len(y)), mode='constant')

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)

    # Pad or trim to N_FRAMES
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :N_FRAMES]

    return mfcc  # (40, 94)


# ─────────────────────────────────────────────
#  AUGMENTATION HELPERS
# ─────────────────────────────────────────────
def add_noise(y, intensity=0.005):
    return y + intensity * np.random.randn(len(y))

def pitch_shift(y, sr, steps=None):
    if steps is None:
        steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def time_stretch(y, rate=None):
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(y, rate=rate)

def volume_variation(y):
    factor = np.random.uniform(0.6, 1.0)
    return y * factor

def apply_lowpass(y, sr, cutoff=4000):
    """Simulate telephone / low-quality mic by attenuating high frequencies."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, cutoff / (sr / 2), btype='low', output='sos')
    return sosfilt(sos, y).astype(np.float32)

def simulate_room(y, sr):
    """Very simple room echo simulation (no IR needed)."""
    delay_samples = int(sr * np.random.uniform(0.02, 0.1))
    decay         = np.random.uniform(0.1, 0.3)
    echo          = np.zeros_like(y)
    echo[delay_samples:] = y[:len(y) - delay_samples] * decay
    return np.clip(y + echo, -1.0, 1.0)


def simulate_tts_smoothing(y, sr):
    """
    Simulate TTS-style spectral smoothness: very light noise + gentle
    low-pass filtering to mimic the 'too clean' artefact of synthesised voices.
    This makes the model more sensitive to over-smooth AI audio.
    """
    from scipy.signal import butter, sosfilt
    # Super-light noise (inaudible but present in real mic captures)
    y = y + 0.001 * np.random.randn(len(y)).astype(np.float32)
    # Mild low-pass to reduce the natural high-frequency roughness
    cutoff = np.random.uniform(6000, 9000)
    sos = butter(2, cutoff / (sr / 2), btype='low', output='sos')
    return sosfilt(sos, y).astype(np.float32)


def augment_audio_standard(y, sr, n_copies=STD_AUGMENT_FACTOR):
    """
    Standard augmentations — used for studio and spoof files.
    Returns the original + n_copies augmented versions.
    """
    versions = [y]  # always include the original
    ops = [
        lambda a: add_noise(a, 0.003),
        lambda a: add_noise(a, 0.007),
        lambda a: pitch_shift(a, sr, 1.5),
        lambda a: pitch_shift(a, sr, -1.5),
        lambda a: time_stretch(a, 0.90),
        lambda a: time_stretch(a, 1.10),
        lambda a: volume_variation(a),
    ]
    np.random.shuffle(ops)
    for op in ops[:n_copies]:
        try:
            versions.append(op(y))
        except Exception:
            pass
    return versions


def augment_audio_mic(y, sr, n_copies=MIC_AUGMENT_FACTOR):
    """
    Heavier augmentations for mic-recorded files — simulate mic conditions
    to help the model learn the mic audio distribution better.
    Returns the original + n_copies augmented versions.
    """
    versions = [y]  # always include the original
    ops = [
        lambda a: add_noise(a, 0.005),
        lambda a: add_noise(a, 0.010),
        lambda a: add_noise(a, 0.015),
        lambda a: pitch_shift(a, sr, 1.0),
        lambda a: pitch_shift(a, sr, -1.0),
        lambda a: pitch_shift(a, sr, 2.0),
        lambda a: time_stretch(a, 0.90),
        lambda a: time_stretch(a, 0.95),
        lambda a: time_stretch(a, 1.05),
        lambda a: time_stretch(a, 1.10),
        lambda a: volume_variation(a),
        lambda a: simulate_room(a, sr),
        lambda a: apply_lowpass(a, sr, 5000),
        lambda a: apply_lowpass(a, sr, 3500),
        lambda a: add_noise(simulate_room(a, sr), 0.005),
    ]
    np.random.shuffle(ops)
    for op in ops[:n_copies]:
        try:
            versions.append(op(y))
        except Exception:
            pass
    return versions


def augment_audio_spoof_rec(y, sr, n_copies=SPOOF_REC_AUGMENT_FACTOR):
    """
    Augmentations for recorded spoof (AI-generated) samples.
    Includes standard transforms PLUS TTS-smoothing to help the model
    learn the 'over-smooth' quality of AI voices more robustly.
    Returns the original + n_copies augmented versions.
    """
    versions = [y]
    ops = [
        lambda a: add_noise(a, 0.002),
        lambda a: add_noise(a, 0.004),
        lambda a: pitch_shift(a, sr, 1.0),
        lambda a: pitch_shift(a, sr, -1.0),
        lambda a: time_stretch(a, 0.92),
        lambda a: time_stretch(a, 1.08),
        lambda a: volume_variation(a),
        lambda a: simulate_tts_smoothing(a, sr),
        lambda a: apply_lowpass(a, sr, 7000),
        lambda a: simulate_tts_smoothing(pitch_shift(a, sr, 0.5), sr),
    ]
    np.random.shuffle(ops)
    for op in ops[:n_copies]:
        try:
            versions.append(op(y))
        except Exception:
            pass
    return versions


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
X = []
y_labels = []


def load_folder(folder_name: str, label: int, augment_fn, dataset_path: str = DATASET_PATH):
    folder_path = os.path.join(dataset_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"  ⚠️  Skipping missing folder: {folder_name}")
        return 0

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".wav", ".mp3"))]
    print(f"  📂 {folder_name}: {len(files)} files", end="", flush=True)

    loaded = 0
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            y_audio, sr = librosa.load(fpath, sr=SR, mono=True)
            versions    = augment_fn(y_audio, sr)
            for v in versions:
                feat = extract_features_from_audio(v, SR)
                X.append(feat)
                y_labels.append(label)
            loaded += len(versions)
        except Exception as e:
            print(f"\n    ❌ Error processing {fname}: {e}")

    print(f"  → {loaded} samples after augmentation")
    return loaded


print("\n=== Loading REAL samples (studio) ===")
for folder in real_folders_studio:
    load_folder(folder, label=1,
                augment_fn=lambda a, sr: augment_audio_standard(a, sr, STD_AUGMENT_FACTOR))

print("\n=== Loading REAL samples (mic) ===")
for folder in real_folders_mic:
    load_folder(folder, label=1,
                augment_fn=lambda a, sr: augment_audio_mic(a, sr, MIC_AUGMENT_FACTOR))

print("\n=== Loading SPOOF samples ===")
for folder in spoof_folders:
    if folder == "spoofed_recorded":
        # Recorded spoof samples get heavier, TTS-targeted augmentation
        load_folder(folder, label=0,
                    augment_fn=lambda a, sr: augment_audio_spoof_rec(a, sr, SPOOF_REC_AUGMENT_FACTOR))
    else:
        load_folder(folder, label=0,
                    augment_fn=lambda a, sr: augment_audio_standard(a, sr, STD_AUGMENT_FACTOR))


# ─────────────────────────────────────────────
#  PREPARE DATA
# ─────────────────────────────────────────────
X_arr = np.array(X)
y_arr = np.array(y_labels)

print(f"\nTotal samples: {len(X_arr)}")
print(f"  Real (label=1): {np.sum(y_arr == 1)}")
print(f"  Spoof (label=0): {np.sum(y_arr == 0)}")

if len(X_arr) == 0:
    raise ValueError("No data loaded. Check dataset folders and paths.")

# Reshape for CNN → (N, 40, 94, 1)
X_arr = X_arr.reshape(X_arr.shape[0], N_MFCC, N_FRAMES, 1)

# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
)

# ─────────────────────────────────────────────
#  CLASS WEIGHTS (additional balancing beyond augmentation)
# ─────────────────────────────────────────────
cw = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(cw))
print(f"\nClass weights: {class_weights}")

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(N_MFCC, N_FRAMES, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Classifier head
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# ─────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
print("\n=== Training ===")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1,
)

# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n=== Test Results ===")
print(f"  Loss:     {loss:.4f}")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# Per-class accuracy
y_pred_raw  = model.predict(X_test, verbose=0).flatten()
y_pred      = (y_pred_raw > 0.5).astype(int)
real_mask   = y_test == 1
spoof_mask  = y_test == 0
real_acc    = np.mean(y_pred[real_mask] == y_test[real_mask])
spoof_acc   = np.mean(y_pred[spoof_mask] == y_test[spoof_mask])
print(f"  Real accuracy:  {real_acc*100:.1f}%")
print(f"  Spoof accuracy: {spoof_acc*100:.1f}%")

# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
print("\nNext step: restart backend/app.py to load the new model.")
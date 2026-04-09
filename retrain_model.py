import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# FIXED DATASET PATH
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")

MODEL_PATH = os.path.join(BASE_DIR, "model", "voiceguard_model.keras")

print("Dataset Path:", DATASET_PATH)
# ===============================
# AUDIO CONFIG
# ===============================

SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40

# Real folders
REAL_FOLDERS = [
    "real_samples",
    "real_mic_recorded"
]

# Spoof folders
SPOOF_FOLDERS = [
    "FlashSpeech",
    "NaturalSpeech3",
    "OpenAI",
    "PromptTTS2",
    "seedtts_files",
    "VALLE",
    "VoiceBox",
    "xTTS",
    "spoofed_recorded"
]

# ===============================
# FEATURE EXTRACTION
# ===============================

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        audio, _ = librosa.effects.trim(audio, top_db=30)

        target_len = SAMPLE_RATE * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        audio = librosa.util.normalize(audio)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC
        )

        mfcc = np.expand_dims(mfcc, axis=-1)

        return mfcc

    except Exception as e:
        print("Error processing:", file_path)
        return None


# ===============================
# LOAD DATA
# ===============================

X = []
y = []

def load_folder(folder_name, label):
    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.exists(folder_path):
        print("Folder not found:", folder_path)
        return

    print("Loading:", folder_path)

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                file_path = os.path.join(root, file)
                features = extract_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(label)

# Load real voices
for folder in REAL_FOLDERS:
    load_folder(folder, 0)

# Load spoof voices
for folder in SPOOF_FOLDERS:
    load_folder(folder, 1)

X = np.array(X)
y = np.array(y)

print("\nTotal Samples:", len(X))
print("Real:", np.sum(y == 0))
print("Spoof:", np.sum(y == 1))

if len(X) == 0:
    raise ValueError("No audio files found. Check dataset path.")

# ===============================
# SPLIT DATA
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Balance classes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

# ===============================
# BUILD CNN MODEL
# ===============================

input_shape = X_train.shape[1:]

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_test, y_test)
print("\nFinal Accuracy:", acc)

# ===============================
# SAVE MODEL
# ===============================

os.makedirs("model", exist_ok=True)
model.save("model/voice_model.keras")
print("\nModel saved at:", MODEL_PATH)
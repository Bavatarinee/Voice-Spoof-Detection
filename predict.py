import numpy as np
import librosa
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "voice_model.keras")

model = load_model(MODEL_PATH)

THRESHOLD = 0.4   # Start with 0.4 for better real detection


def extract_features(file_path):
    y_audio, sr = librosa.load(file_path, sr=22050)
    y_audio = librosa.util.normalize(y_audio)

    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)

    # Pad or trim to 94 frames
    if mfcc.shape[1] < 94:
        pad_width = 94 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :94]

    mfcc = mfcc.reshape(1, 40, 94, 1)

    return mfcc


def predict_voice(file_path):
    features = extract_features(file_path)

    prediction = model.predict(features)[0][0]

    print("Raw prediction:", prediction)

    if prediction > THRESHOLD:
        result = "Real Voice"
        confidence = prediction
    else:
        result = "Spoofed Voice"
        confidence = 1 - prediction

    return result, round(float(confidence) * 100, 2)
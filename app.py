import os
import uuid
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans


REAL_THRESHOLD = 0.35     # Lowered from 0.42 to account for post-retraining strictness
SPEECH_RMS_MIN = 0.015    # Slightly increased to filter out pure hiss


WINDOW_SECONDS   = 3      # each analysis window is 3 s
WINDOW_OVERLAP   = 0.5    # 50% overlap between windows
SR               = 22050  # must match training


MIN_ACTIVE_WINDOWS = 1


REAL_VOTE_RATIO = 0.35

app = Flask(__name__)
CORS(app) # Allow cross-origin requests from the mobile app

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH    = os.path.join(BASE_DIR, "model", "voice_model.keras")
REAL_MIC_FOLDER = os.path.join(
    os.path.dirname(BASE_DIR), "dataset", "real_mic_recorded"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")
print("Expected input shape:", model.input_shape)
EXPECTED_CHANNELS = model.input_shape[-1]


def extract_features_from_audio(y: np.ndarray, sr: int = SR, channels: int = 3) -> np.ndarray:
    """
    Extract MFCC features + optional Delta + Delta-Delta.
    Matches the training pipeline requirements dynamically.
    """
    # 1. Resample if needed
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # 2. Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)

    # 3. Ensure minimum 1 second of audio
    if len(y) < SR:
        y = np.pad(y, (0, SR - len(y)), mode='constant')

    # 4. Extract 40 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)

    # 5. Pad or trim to exactly 94 frames
    if mfcc.shape[1] < 94:
        mfcc = np.pad(mfcc, ((0, 0), (0, 94 - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :94]

    if channels == 3:
        # Delta and Delta-Delta MFCCs
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        # Stack into 3 channels (40, 94, 3)
        features = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
    else:
        # Single channel (40, 94, 1)
        features = mfcc[..., np.newaxis]

    return features


def extract_features(file_path: str) -> np.ndarray:
    """Load audio file and extract features."""
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    return extract_features_from_audio(y, sr)


def spoof_penalty(y: np.ndarray, sr: int = SR) -> float:
    """
    Rule-based spoof confidence penalty.
    Returns a value in [0.0, 0.20] that is subtracted from the CNN score.
    0.0 = no penalty (looks natural), 0.20 = strong spoof signal.

    CALIBRATION NOTE:
    Thresholds have been loosened compared to earlier versions because
    live microphone recordings have naturally lower spectral flux and ZCR
    variance than studio-quality audio — the mic compresses dynamics and
    adds a mild low-pass coloring that makes real voices look 'smooth'.
    Only penalise characteristics that are far outside any plausible
    real-mic distribution.
    """
    penalty = 0.0

   
    try:
        S = np.abs(librosa.stft(y))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        flux_cv = float(np.std(flux) / (np.mean(flux) + 1e-9))
        print(f"[DEBUG] Spectral flux CV: {flux_cv:.4f}")
        if flux_cv < 0.22:      # Only flag extremely smooth (AI-like) signals
            penalty += 0.10
        elif flux_cv < 0.33:
            penalty += 0.04
    except Exception:
        pass

    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_std = float(np.std(zcr))
        print(f"[DEBUG] ZCR std: {zcr_std:.5f}")
        if zcr_std < 0.012:     # Extremely low — strong AI synthesiser signal
            penalty += 0.07
        elif zcr_std < 0.020:
            penalty += 0.03
    except Exception:
        pass

    try:
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        mean_flat = float(np.mean(flatness))
        print(f"[DEBUG] Spectral flatness: {mean_flat:.5f}")
        # Only penalise truly extreme values unlikely for any real mic recording
        if mean_flat > 0.22 or mean_flat < 0.001:
            penalty += 0.05
    except Exception:
        pass

    return min(penalty, 0.20)   # cap at 0.20 (was 0.30 — too harsh for mic audio)

def predict_with_voting(y: np.ndarray, sr: int = SR) -> dict:
    """
    Run model on multiple overlapping windows and aggregate scores.

    Key logic:
      1. Compute RMS energy of each window.
      2. Skip windows below SPEECH_RMS_MIN (silence / breath sounds).
      3. Compute the TRIMMED MEAN of active-window CNN scores.
      4. Apply per-window rule-based spoof_penalty on the whole clip.
      5. Fall back to all windows if too few active ones are found.
      6. Require REAL_VOTE_RATIO of windows to vote REAL (majority rule)
         as an additional guard against isolated high-scoring windows
         in otherwise spoof recordings.

    NOTE: The REAL_RESCUE shortcut has been intentionally removed because
    it allowed AI voices scoring high on a single window to bypass the
    threshold, leading to false REAL decisions.

    Returns: {
        "raw_score"      : float,          # final aggregated score (after penalty)
        "cnn_score"      : float,          # CNN-only score before penalty
        "penalty"        : float,          # rule-based penalty applied
        "window_scores"  : list[float],    # all window CNN scores
        "active_scores"  : list[float],    # scores of speech-active windows only
        "real_votes"     : int,            # windows that voted REAL
        "windows_used"   : int,
    }
    """
    window_samples = int(WINDOW_SECONDS * sr)
    hop_samples    = int(window_samples * (1 - WINDOW_OVERLAP))

    # Build window start positions
    starts = []
    pos = 0
    while pos + window_samples <= len(y):
        starts.append(pos)
        pos += hop_samples

    # If audio is shorter than one window, use the whole clip
    if not starts:
        starts = [0]
        y = np.pad(y, (0, max(0, window_samples - len(y))), mode='constant')

    scores   = []
    energies = []
    profiles = [] # Collect speaker profiles (average MFCC per window)
    for start in starts:
        chunk = y[start: start + window_samples]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode='constant')

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        energies.append(rms)

        features = extract_features_from_audio(chunk, sr, channels=EXPECTED_CHANNELS)
        # Use only the base MFCC channel for speaker profile (40-dim)
        profiles.append(np.mean(features[:, :, 0], axis=1)) 
        
        inp      = features.reshape(1, 40, 94, EXPECTED_CHANNELS)
        score    = float(model.predict(inp, verbose=0)[0][0])
        scores.append(score)

    # ── Debug printout ────────────────────────────────────────────────
    for i, (s, e) in enumerate(zip(scores, energies)):
        active = e >= SPEECH_RMS_MIN
        tag = "[SPEECH]" if active else "[SILENT]"
        print(f"  Window {i+1}: score={s:.4f}  rms={e:.5f}  {tag}")

    # ── Filter: keep only speech-active windows ───────────────────────
    active_indices = [i for i, e in enumerate(energies) if e >= SPEECH_RMS_MIN]
    
    if len(active_indices) < MIN_ACTIVE_WINDOWS:
        # Not enough speech detected — fall back to all windows
        active_indices = list(range(len(scores)))
        print(f"[DEBUG] Too few active windows, using ALL {len(active_indices)} windows")

    active_scores   = [scores[i] for i in active_indices]
    active_profiles = [profiles[i] for i in active_indices]

    # ── Speaker Count Heuristic ───────────────────────────────────────
    speaker_count = 1
    if len(active_profiles) >= 3:
        # Use KMeans to see if the windows cluster into 2 distinct speakers
        X = np.array(active_profiles)
        # Standardize for clustering
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        
        inertia_1 = np.sum((X - np.mean(X, axis=0))**2)
        km2 = KMeans(n_clusters=2, n_init=10, random_state=42)
        km2.fit(X)
        inertia_2 = km2.inertia_
        
        # If reduction in variance is > 55%, it's likely multiple speakers
        reduction = (inertia_1 - inertia_2) / (inertia_1 + 1e-9)
        print(f"[DEBUG] Speaker clustering reduction: {reduction:.2%}")
        if reduction > 0.55:
            speaker_count = 2
    elif len(active_profiles) == 2:
        # Just compare the two windows
        dist = np.linalg.norm(active_profiles[0] - active_profiles[1])
        if dist > 15.0: # Heuristic distance for MFCC averages
            speaker_count = 2

    # ── Robust mean (drop the lowest 30% to ignore silence and top 10% outliers) ──
    arr = np.sort(active_scores)
    drop_bottom = min(int(len(arr) * 0.30), len(arr) - 1)
    drop_top = min(int(len(arr) * 0.10), len(arr) - 1 - drop_bottom)
    
    if len(arr) > (drop_bottom + drop_top):
        arr = arr[drop_bottom:len(arr)-drop_top]
        
    cnn_score = float(np.mean(arr))

    # ── Rule-based spoof penalty on the full clip ─────────────────────
    penalty = spoof_penalty(y, sr)
    final_score = max(0.0, cnn_score - penalty)

    # ── Majority vote guard ───────────────────────────────────────────
    real_votes = sum(1 for s in active_scores if s > REAL_THRESHOLD)
    vote_ratio = real_votes / max(len(active_scores), 1)
    
    if speaker_count == 1:
        if vote_ratio > 0.6:
            # Very likely to be REAL if majority windows strongly agree
            final_score = max(final_score, REAL_THRESHOLD + 0.05)
        # We NO LONGER instantly penalize if vote_ratio <= 0.5, because
        # trailing silence often creates a 50/50 split naturally for live mics.

    # Remove the OLD strict override entirely
    # if vote_ratio < REAL_VOTE_RATIO and len(active_scores) >= 2: ...

    print(f"[DEBUG] Speaker Count:      {speaker_count}")
    print(f"[DEBUG] CNN (trimmed mean): {cnn_score:.4f}")
    print(f"[DEBUG] Spoof penalty:      {penalty:.4f}")
    print(f"[DEBUG] Final score:        {final_score:.4f} | Threshold: {REAL_THRESHOLD}")
    print(f"[DEBUG] Real votes:         {real_votes}/{len(active_scores)} ({vote_ratio:.0%})")

    return {
        "raw_score":     final_score,
        "cnn_score":     cnn_score,
        "penalty":       penalty,
        "window_scores": scores,
        "active_scores": active_scores,
        "real_votes":    real_votes,
        "windows_used":  len(scores),
        "speaker_count": speaker_count,
    }


def compute_confidence(raw_score: float, is_real: bool) -> float:
    """
    Map raw_score distance from threshold to a calibrated confidence (50–99%).
    """
    dist = abs(raw_score - REAL_THRESHOLD)
    # Sigmoid-like mapping: dist=0 → 50%, dist=0.5+ → ~99%
    conf = 0.50 + 0.49 * (1 - np.exp(-dist * 8))
    return round(float(conf) * 100, 1)

@app.route("/ping", methods=["GET"])
def ping():
    """Lightweight health-check for the mobile app connection test."""
    return jsonify({"status": "ok", "threshold": REAL_THRESHOLD})


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    safe_name = f"audio_{uuid.uuid4().hex}.wav"
    filepath  = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)

    try:
        # Load audio once
        y, sr = librosa.load(filepath, sr=SR, mono=True)

        # Use multi-window voting for robustness
        vote_result  = predict_with_voting(y, sr)
        raw_score    = vote_result["raw_score"]
        window_scores = vote_result["window_scores"]
        windows_used  = vote_result["windows_used"]

        vote_ratio = vote_result.get("real_votes", 0) / max(len(vote_result.get("active_scores", [])), 1)
        is_real    = raw_score > REAL_THRESHOLD

        if vote_result.get("speaker_count", 1) >= 2 and 0.15 <= vote_ratio <= 0.85:
            human_pct = int(round(vote_ratio * 100))
            result = f"{human_pct}% HUMAN, {100 - human_pct}% AI"
        else:
            result = "REAL VOICE" if is_real else "AI VOICE"
            
        confidence = compute_confidence(raw_score, is_real)

        return jsonify({
            "result":        result,
            "confidence":    confidence,
            "raw_score":     round(raw_score, 4),
            "cnn_score":     round(vote_result.get("cnn_score", raw_score), 4),
            "penalty":       round(vote_result.get("penalty", 0.0), 4),
            "window_scores": [round(s, 4) for s in vote_result["window_scores"]],
            "active_scores": [round(s, 4) for s in vote_result.get("active_scores", [])],
            "real_votes":    vote_result.get("real_votes", 0),
            "windows_used":  vote_result["windows_used"],
            "threshold":     REAL_THRESHOLD,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


@app.route("/save-real", methods=["POST"])
def save_real_sample():
    """
    Save a recorded WAV into the real_mic_recorded dataset folder.
    Use the UI's 'Save as Real Training Sample' button after recording.
    After saving 50+ new samples, retrain the model.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    os.makedirs(REAL_MIC_FOLDER, exist_ok=True)

    existing  = [f for f in os.listdir(REAL_MIC_FOLDER)
                 if f.lower().endswith(('.wav', '.mp3'))]
    next_num  = len(existing) + 1
    save_name = f"mic_{next_num:04d}.wav"
    save_path = os.path.join(REAL_MIC_FOLDER, save_name)
    file.save(save_path)

    total = len(existing) + 1
    print(f"[DATASET] Saved real mic sample: {save_name} (total: {total})")
    return jsonify({
        "message":          f"Saved as {save_name}",
        "total_mic_samples": total,
    })

SPOOF_REC_FOLDER = os.path.join(
    os.path.dirname(BASE_DIR), "dataset", "spoofed_recorded"
)

@app.route("/save-spoof", methods=["POST"])
def save_spoof_sample():
    """
    Save a recorded WAV as a confirmed spoof sample for retraining.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    os.makedirs(SPOOF_REC_FOLDER, exist_ok=True)

    existing  = [f for f in os.listdir(SPOOF_REC_FOLDER)
                 if f.lower().endswith(('.wav', '.mp3'))]
    next_num  = len(existing) + 1
    save_name = f"spoof_{next_num:04d}.wav"
    save_path = os.path.join(SPOOF_REC_FOLDER, save_name)
    file.save(save_path)

    total = len(existing) + 1
    print(f"[DATASET] Saved spoof sample: {save_name} (total: {total})")
    return jsonify({
        "message":             f"Saved as {save_name}",
        "total_spoof_samples": total,
    })



@app.route("/debug-score", methods=["POST"])
def debug_score():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file      = request.files["file"]
    safe_name = f"dbg_{uuid.uuid4().hex}.wav"
    filepath  = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)

    try:
        y, sr    = librosa.load(filepath, sr=SR, mono=True)
        result   = predict_with_voting(y, sr)
        raw      = result["raw_score"]
        
        vote_ratio = result.get("real_votes", 0) / max(len(result.get("active_scores", [])), 1)
        
        # Consistent logic with /predict: only show percentages if multiple potential speakers detected
        if result.get("speaker_count", 1) >= 2 and 0.15 <= vote_ratio <= 0.85:
            human_pct = int(round(vote_ratio * 100))
            decision = f"{human_pct}% HUMAN, {100 - human_pct}% AI"
        else:
            decision = "REAL VOICE" if raw > REAL_THRESHOLD else "AI VOICE"

        # Also compute single-window score for comparison
        features = extract_features_from_audio(y, SR, channels=EXPECTED_CHANNELS)
        inp      = features.reshape(1, 40, 94, EXPECTED_CHANNELS)
        single   = float(model.predict(inp, verbose=0)[0][0])

        return jsonify({
            "single_window_score": round(single, 6),
            "cnn_score":           round(result.get("cnn_score", raw), 6),
            "spoof_penalty":       round(result.get("penalty", 0.0), 6),
            "final_score":         round(raw, 6),
            "real_votes":          result.get("real_votes", 0),
            "windows_used":        result["windows_used"],
            "window_scores":       [round(s, 4) for s in result["window_scores"]],
            "active_scores":       [round(s, 4) for s in result.get("active_scores", [])],
            "threshold":           REAL_THRESHOLD,
            "decision":            decision,
            "note": (
                f"Score > {REAL_THRESHOLD} = REAL. "
                "CNN score is raw model output; penalty is subtracted for spectral spoof cues. "
                "Use 'Save as Spoof' to add more AI voice samples and retrain."
            )
        })
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        try:
            os.remove(filepath)
        except Exception:
            pass



if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print("\n" + "="*60)
    print("  Voice Spoof Detection Server")
    print("="*60)
    print(f"  Local:   https://127.0.0.1:5000")
    print(f"  Network: https://{local_ip}:5000")
    print(f"  Threshold: {REAL_THRESHOLD}")
    print("="*60 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True,
        ssl_context='adhoc',   # Self-signed HTTPS — required for iOS mic access
    )
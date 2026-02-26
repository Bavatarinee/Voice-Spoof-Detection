import os
import uuid
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
#  THRESHOLDS
#
#  REAL_THRESHOLD   – aggregated window score must EXCEED this to be REAL.
#                     Raised to 0.50 so the model must be meaningfully
#                     confident a voice is real before accepting it.
#                     Lower values let AI-generated voices slip through.
#
#  SPEECH_RMS_MIN   – windows whose RMS energy is below this are
#                     treated as silence and excluded from voting.
#
#  NOTE: The REAL_RESCUE shortcut has been REMOVED.  It caused AI voices
#        that scored high on even one window to bypass the threshold check
#        and be accepted as real.  Every window must now earn its verdict.
# ─────────────────────────────────────────────
REAL_THRESHOLD = 0.50
SPEECH_RMS_MIN = 0.02    # ignore windows quieter than this

# How many windows to slice from a long recording for voting
WINDOW_SECONDS   = 3      # each analysis window is 3 s
WINDOW_OVERLAP   = 0.5    # 50% overlap between windows
SR               = 22050  # must match training

# Minimum number of active (non-silent) windows required.
# If fewer speech windows are found, fall back to ALL windows.
MIN_ACTIVE_WINDOWS = 1

# Fraction of active windows that MUST vote REAL for the clip to be REAL.
# 0.55 means at least 55 % of voiced windows must score > REAL_THRESHOLD.
REAL_VOTE_RATIO = 0.55

app = Flask(__name__)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH    = os.path.join(BASE_DIR, "model", "voice_model.keras")
REAL_MIC_FOLDER = os.path.join(
    os.path.dirname(BASE_DIR), "dataset", "real_mic_recorded"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")
print("Expected input shape:", model.input_shape)


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
#
#  ⚠️  MUST EXACTLY MATCH train_model.py ⚠️
#
#  Training pipeline (train_model.py):
#    1. librosa.load(…, sr=22050)
#    2. librosa.util.normalize
#    3. NO trim  ← training did NOT trim
#    4. mfcc(n_mfcc=40)
#    5. pad/clip to 94 frames
#
#  This function now matches that exactly.
# ─────────────────────────────────────────────
def extract_features_from_audio(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extract MFCC features from a pre-loaded audio array.
    Matches the training pipeline in train_model.py exactly.
    """
    # 1. Resample if needed (should already be SR from load)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR

    # 2. Normalize amplitude (same as training)
    if np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)

    # 3. Ensure minimum 1 second of audio (pad if too short)
    min_samples = SR  # 1 second
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)), mode='constant')

    # 4. Extract 40 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)

    # 5. Pad or trim to exactly 94 frames
    if mfcc.shape[1] < 94:
        mfcc = np.pad(mfcc, ((0, 0), (0, 94 - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :94]

    return mfcc  # shape (40, 94)


def extract_features(file_path: str) -> np.ndarray:
    """Load audio file and extract features."""
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    return extract_features_from_audio(y, sr)


# ─────────────────────────────────────────────
#  AUXILIARY SPOOF SCORE  (rule-based, complements the CNN)
#
#  AI-synthesised voices have characteristic properties that MFCCs alone
#  may miss after a model is trained mostly on studio/dataset data:
#
#    • Too-smooth spectral flux  – real voices have natural micro-variation
#    • Very low ZCR variance     – AI speech tends to be steady
#    • High harmonic regularity  – synthesisers produce near-perfect pitch
#
#  This function returns a spoof_penalty in [0, 1].  A higher value means
#  the clip looks MORE like a spoof.  The penalty is subtracted from the
#  CNN score before the threshold check.
# ─────────────────────────────────────────────
def spoof_penalty(y: np.ndarray, sr: int = SR) -> float:
    """
    Rule-based spoof confidence penalty.
    Returns a value in [0.0, 0.30] that is subtracted from the CNN score.
    0.0 = no penalty (looks natural), 0.30 = strong spoof signal.
    """
    penalty = 0.0

    # — Spectral flux variance (real speech is more irregular) ————————
    try:
        S = np.abs(librosa.stft(y))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        flux_cv = float(np.std(flux) / (np.mean(flux) + 1e-9))   # coeff of variation
        # Real speech typically has flux_cv > 0.6; AI tends to be smoother
        if flux_cv < 0.40:
            penalty += 0.12
        elif flux_cv < 0.55:
            penalty += 0.06
    except Exception:
        pass

    # — ZCR (zero-crossing rate) variance ——————————————————————————
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_std = float(np.std(zcr))
        # Real speech has more variance in ZCR across phonemes
        if zcr_std < 0.025:
            penalty += 0.10
        elif zcr_std < 0.040:
            penalty += 0.05
    except Exception:
        pass

    # — Spectral flatness (closer to 1 = more noise-like / synthesised) ——
    try:
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        mean_flat = float(np.mean(flatness))
        # Extremely flat OR extremely tonal both look synthetic
        if mean_flat > 0.15 or mean_flat < 0.002:
            penalty += 0.08
    except Exception:
        pass

    return min(penalty, 0.30)   # cap at 0.30


# ─────────────────────────────────────────────
#  MULTI-WINDOW PREDICTION
#
#  Slices the recording into overlapping 3-second windows,
#  runs the model on each window, and averages the scores.
#  This greatly improves robustness for live mic recordings
#  compared to a single whole-clip prediction.
# ─────────────────────────────────────────────
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
    for start in starts:
        chunk = y[start: start + window_samples]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode='constant')

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        energies.append(rms)

        features = extract_features_from_audio(chunk, sr)
        inp      = features.reshape(1, 40, 94, 1)
        score    = float(model.predict(inp, verbose=0)[0][0])
        scores.append(score)

    # ── Debug printout ────────────────────────────────────────────────
    for i, (s, e) in enumerate(zip(scores, energies)):
        active = e >= SPEECH_RMS_MIN
        tag = "[SPEECH]" if active else "[SILENT]"
        print(f"  Window {i+1}: score={s:.4f}  rms={e:.5f}  {tag}")

    # ── Filter: keep only speech-active windows ───────────────────────
    active_pairs = [(s, e) for s, e in zip(scores, energies) if e >= SPEECH_RMS_MIN]

    if len(active_pairs) < MIN_ACTIVE_WINDOWS:
        # Not enough speech detected — fall back to all windows
        active_pairs = list(zip(scores, energies))
        print(f"[DEBUG] Too few active windows, using ALL {len(active_pairs)} windows")

    active_scores = [s for s, _ in active_pairs]

    # ── Trimmed mean (drop top & bottom 10% to remove outliers) ───────
    arr = np.sort(active_scores)
    trim = max(1, int(len(arr) * 0.10))
    if len(arr) > 2 * trim:
        arr = arr[trim:-trim]
    cnn_score = float(np.mean(arr))

    # ── Rule-based spoof penalty on the full clip ─────────────────────
    penalty = spoof_penalty(y, sr)
    final_score = max(0.0, cnn_score - penalty)

    # ── Majority vote guard ───────────────────────────────────────────
    real_votes = sum(1 for s in active_scores if s > REAL_THRESHOLD)
    vote_ratio = real_votes / max(len(active_scores), 1)
    # Override to spoof if not enough windows vote REAL
    if vote_ratio < REAL_VOTE_RATIO and len(active_scores) >= 2:
        print(f"[DEBUG] Majority-vote override: only {real_votes}/{len(active_scores)} "
              f"windows voted REAL ({vote_ratio:.0%} < required {REAL_VOTE_RATIO:.0%})")
        # Cap the score below threshold so it registers as spoof
        final_score = min(final_score, REAL_THRESHOLD - 0.01)

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
    }


# ─────────────────────────────────────────────
#  CONFIDENCE CALCULATION
#
#  Instead of the raw model score as confidence,
#  we compute how far the score is from the threshold
#  and map it to a [50%, 99%] range.
#  This prevents the UI from showing "100% confident" or "1% confident"
#  which are misleading for a threshold-based system.
# ─────────────────────────────────────────────
def compute_confidence(raw_score: float, is_real: bool) -> float:
    """
    Map raw_score distance from threshold to a calibrated confidence (50–99%).
    """
    dist = abs(raw_score - REAL_THRESHOLD)
    # Sigmoid-like mapping: dist=0 → 50%, dist=0.5+ → ~99%
    conf = 0.50 + 0.49 * (1 - np.exp(-dist * 8))
    return round(float(conf) * 100, 1)


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
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

        is_real    = raw_score > REAL_THRESHOLD
        result     = "REAL VOICE" if is_real else "SPOOF VOICE"
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


# ─────────────────────────────────────────────
#  SAVE REAL SAMPLE (for retraining)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  SAVE SPOOF SAMPLE (for retraining)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  DEBUG ENDPOINT
# ─────────────────────────────────────────────
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
        decision = "REAL" if raw > REAL_THRESHOLD else "SPOOF"

        # Also compute single-window score for comparison
        features = extract_features_from_audio(y, SR)
        inp      = features.reshape(1, 40, 94, 1)
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


# ─────────────────────────────────────────────
#  RUN SERVER
# ─────────────────────────────────────────────
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
    print("  NOTE: Browser will warn about self-signed cert.")
    print("        Click 'Advanced' → 'Proceed' to continue.")
    print("="*60 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True,
        ssl_context="adhoc",
    )
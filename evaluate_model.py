"""
evaluate_model.py
=================
Voice Spoof Detection — Model Evaluation
-----------------------------------------
Loads the saved Keras model and re-builds the test set from the dataset,
then computes and plots:

  1.  Confusion Matrix (absolute + normalised)
  2.  Classification Report  (precision / recall / F1 / support)
  3.  ROC Curve + AUC
  4.  Precision-Recall Curve + AP
  5.  Per-class accuracy  (Real vs Spoof)
  6.  Overall accuracy, balanced accuracy, MCC
  7.  Training History plots (if history CSV is present)

Outputs are saved to:
    backend/training/evaluation_results/

Usage:
    python evaluate_model.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  (must match train_model.py exactly)
# ─────────────────────────────────────────────────────────────
DATASET_PATH    = r"D:\Dissertion 2\voice spoof try2\dataset"
MODEL_PATH      = r"D:\Dissertion 2\voice spoof try2\backend\model\voice_model.keras"
RESULTS_DIR     = r"D:\Dissertion 2\voice spoof try2\backend\training\evaluation_results"

SR       = 22050
N_MFCC   = 40
N_FRAMES = 94

# THRESHOLD for binary classification (default: 0.5)
THRESHOLD = 0.5

REAL_LABEL  = 1    # 1 = real / genuine voice
SPOOF_LABEL = 0    # 0 = spoofed / AI-generated voice

# Dataset folders (same as train_model.py — no augmentation here, raw samples only)
real_folders_studio = ["real_samples"]
real_folders_mic    = ["real_mic_recorded"]
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

# ─────────────────────────────────────────────────────────────
#  SEABORN / MATPLOTLIB STYLING
# ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
CMAP = "Blues"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION  (must match train_model.py exactly)
# ─────────────────────────────────────────────────────────────
def extract_features_from_audio(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Extract 40-MFCC features — mirrors app.py / train_model.py."""
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)
    if len(y) < SR:
        y = np.pad(y, (0, SR - len(y)), mode="constant")
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :N_FRAMES]
    return mfcc  # shape (40, 94)


# ─────────────────────────────────────────────────────────────
#  DATA LOADING  (raw, no augmentation — evaluation should use
#                 the original audio only)
# ─────────────────────────────────────────────────────────────
def load_folder_raw(folder_name: str, label: int):
    """Load all WAV/MP3 files from a folder without augmentation."""
    samples, labels = [], []
    folder_path = os.path.join(DATASET_PATH, folder_name)
    if not os.path.exists(folder_path):
        print(f"  ⚠️  Skipping missing folder: {folder_name}")
        return samples, labels

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".wav", ".mp3"))]
    print(f"  📂 {folder_name}: {len(files)} files", end="", flush=True)

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            y_audio, sr = librosa.load(fpath, sr=SR, mono=True)
            feat = extract_features_from_audio(y_audio, sr)
            samples.append(feat)
            labels.append(label)
        except Exception as e:
            print(f"\n    ❌ Error: {fname}: {e}")

    print(f"  → {len(samples)} samples")
    return samples, labels


def build_dataset():
    """Build the full raw dataset (no augmentation)."""
    X, y_labels = [], []

    print("\n=== Loading REAL samples (studio) ===")
    for folder in real_folders_studio:
        s, l = load_folder_raw(folder, REAL_LABEL)
        X.extend(s); y_labels.extend(l)

    print("\n=== Loading REAL samples (mic) ===")
    for folder in real_folders_mic:
        s, l = load_folder_raw(folder, REAL_LABEL)
        X.extend(s); y_labels.extend(l)

    print("\n=== Loading SPOOF samples ===")
    for folder in spoof_folders:
        s, l = load_folder_raw(folder, SPOOF_LABEL)
        X.extend(s); y_labels.extend(l)

    X_arr = np.array(X).reshape(-1, N_MFCC, N_FRAMES, 1)
    y_arr = np.array(y_labels)
    return X_arr, y_arr


# ─────────────────────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────────────────────
def save(fig, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved: {path}")


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    """Plot both raw-count and normalised confusion matrices side-by-side."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Absolute Counts", "Normalised (row %)"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=CMAP,
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(subtitle, fontsize=12)

    plt.tight_layout()
    save(fig, "confusion_matrix.png")


def plot_roc(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # find the threshold closest to our operating point
    op_idx = np.argmin(np.abs(thresholds - THRESHOLD))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.scatter(fpr[op_idx], tpr[op_idx], s=120, zorder=5, color="#F44336",
               label=f"Threshold = {THRESHOLD:.2f}")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    save(fig, "roc_curve.png")
    return roc_auc


def plot_precision_recall(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(recall, precision, where="post", color="#4CAF50", lw=2.5,
            label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color="#4CAF50", step="post")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    save(fig, "precision_recall_curve.png")
    return ap


def plot_score_distribution(y_true, y_scores):
    """Histogram of model output scores split by class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 40)

    ax.hist(y_scores[y_true == REAL_LABEL],  bins=bins, alpha=0.65,
            color="#2196F3", label="Real Voice",    edgecolor="white")
    ax.hist(y_scores[y_true == SPOOF_LABEL], bins=bins, alpha=0.65,
            color="#F44336", label="Spoofed Voice", edgecolor="white")
    ax.axvline(THRESHOLD, color="black", linestyle="--", lw=1.8,
               label=f"Threshold = {THRESHOLD:.2f}")

    ax.set_xlabel("Model Output Score  (closer to 1 → Real)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score Distribution by Class", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, "score_distribution.png")


def plot_per_class_bar(y_true, y_pred):
    """Bar chart: per-class accuracy."""
    real_mask  = y_true == REAL_LABEL
    spoof_mask = y_true == SPOOF_LABEL
    real_acc   = np.mean(y_pred[real_mask]  == y_true[real_mask])  * 100
    spoof_acc  = np.mean(y_pred[spoof_mask] == y_true[spoof_mask]) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Real Voice", "Spoofed Voice"],
                  [real_acc, spoof_acc],
                  color=["#2196F3", "#F44336"], width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, [real_acc, spoof_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "per_class_accuracy.png")
    return real_acc, spoof_acc


def plot_metrics_summary(metrics: dict):
    """Horizontal bar chart of key scalar metrics."""
    names  = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=1.0)

    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=11)

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Model Performance Summary", fontsize=14, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "metrics_summary.png")


# ─────────────────────────────────────────────────────────────
#  MAIN EVALUATION ROUTINE
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Voice Spoof Detection — Model Evaluation")
    print("=" * 60)

    # ── 1. Verify paths ──────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}\n"
                                "Train the model first with train_model.py")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    # ── 2. Load model ────────────────────────────────────────
    print(f"\n📦 Loading model from:\n   {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    model.summary()

    # ── 3. Build dataset (raw, no augmentation) ──────────────
    print("\n📝 Building evaluation dataset (no augmentation)…")
    X_all, y_all = build_dataset()

    if len(X_all) == 0:
        raise ValueError("No data loaded. Check dataset paths.")

    print(f"\nTotal raw samples: {len(X_all)}")
    print(f"  Real  (label=1): {np.sum(y_all == REAL_LABEL)}")
    print(f"  Spoof (label=0): {np.sum(y_all == SPOOF_LABEL)}")

    # ── 4. Train/test split (same seed as train_model.py) ────
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"\nTest split: {len(X_test)} samples")

    # ── 5. Predict ───────────────────────────────────────────
    print("\n🔍 Running inference…")
    y_scores = model.predict(X_test, batch_size=32, verbose=1).flatten()
    y_pred   = (y_scores >= THRESHOLD).astype(int)

    # ── 6. Scalar metrics ────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy          = (tp + tn) / (tp + tn + fp + fn)
    balanced_acc      = balanced_accuracy_score(y_test, y_pred)
    mcc               = matthews_corrcoef(y_test, y_pred)
    sensitivity       = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / True Positive Rate
    specificity       = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # True Negative Rate
    precision_val     = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1                = (2 * precision_val * sensitivity /
                         (precision_val + sensitivity)
                         if (precision_val + sensitivity) > 0 else 0.0)
    fpr_val           = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Positive Rate
    fnr_val           = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # False Negative Rate

    roc_auc = plot_roc(y_test, y_scores)
    ap      = plot_precision_recall(y_test, y_scores)

    # ── 7. Console report ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Accuracy':<30} {accuracy:>10.4f}  ({accuracy*100:.2f}%)")
    print(f"{'Balanced Accuracy':<30} {balanced_acc:>10.4f}")
    print(f"{'Matthews Corr. Coeff.':<30} {mcc:>10.4f}")
    print(f"{'ROC AUC':<30} {roc_auc:>10.4f}")
    print(f"{'Average Precision (AP)':<30} {ap:>10.4f}")
    print(f"{'Precision (PPV)':<30} {precision_val:>10.4f}")
    print(f"{'Recall / Sensitivity (TPR)':<30} {sensitivity:>10.4f}")
    print(f"{'Specificity (TNR)':<30} {specificity:>10.4f}")
    print(f"{'F1 Score':<30} {f1:>10.4f}")
    print(f"{'False Positive Rate (FPR)':<30} {fpr_val:>10.4f}")
    print(f"{'False Negative Rate (FNR)':<30} {fnr_val:>10.4f}")
    print(f"\n{'Confusion Matrix':}")
    print(f"  {'':15} {'Pred Spoof':>12} {'Pred Real':>12}")
    print(f"  {'True Spoof':15} {tn:>12d} {fp:>12d}")
    print(f"  {'True Real':15} {fn:>12d} {tp:>12d}")

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(
        y_test, y_pred,
        target_names=["Spoofed Voice", "Real Voice"],
        digits=4,
    ))

    # ── 8. Generate all plots ────────────────────────────────
    print("\n📊 Generating plots…")
    plot_confusion_matrix(cm,
                          classes=["Spoofed", "Real"],
                          title="Voice Spoof Detection — Confusion Matrix")
    plot_score_distribution(y_test, y_scores)
    real_acc, spoof_acc = plot_per_class_bar(y_test, y_pred)

    summary_metrics = {
        "Accuracy":           accuracy,
        "Balanced Accuracy":  balanced_acc,
        "ROC AUC":            roc_auc,
        "Average Precision":  ap,
        "F1 Score":           f1,
        "Precision (PPV)":    precision_val,
        "Recall / TPR":       sensitivity,
        "Specificity / TNR":  specificity,
        "MCC":                (mcc + 1) / 2,   # scale MCC [-1,1] → [0,1] for chart
    }
    plot_metrics_summary(summary_metrics)

    # ── 9. Save text report ──────────────────────────────────
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Voice Spoof Detection — Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model : {MODEL_PATH}\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")
        f.write(f"{'Metric':<30} {'Value':>10}\n")
        f.write("-" * 42 + "\n")
        for name, val in summary_metrics.items():
            f.write(f"{name:<30} {val:>10.4f}\n")
        f.write(f"\n{'Accuracy':<30} {accuracy:>10.4f}\n")
        f.write(f"{'Balanced Accuracy':<30} {balanced_acc:>10.4f}\n")
        f.write(f"{'MCC':<30} {mcc:>10.4f}\n")
        f.write(f"{'FPR':<30} {fpr_val:>10.4f}\n")
        f.write(f"{'FNR':<30} {fnr_val:>10.4f}\n")
        f.write(f"\nPer-class Accuracy:\n")
        f.write(f"  Real Voice  : {real_acc:.2f}%\n")
        f.write(f"  Spoof Voice : {spoof_acc:.2f}%\n")
        f.write("\nConfusion Matrix (rows=True, cols=Pred):\n")
        f.write(f"  TN={tn}  FP={fp}\n")
        f.write(f"  FN={fn}  TP={tp}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(
            y_test, y_pred,
            target_names=["Spoofed Voice", "Real Voice"],
            digits=4,
        ))
    print(f"  💾 Saved: {report_path}")

    print(f"\n✅ Evaluation complete! All outputs in:\n   {RESULTS_DIR}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

# ✨ Hybrid Voice Spoofing Detection System

![Accuracy Badge](https://img.shields.io/badge/Accuracy-99.89%25-brightgreen)
![Python Backend](https://img.shields.io/badge/Backend-Flask%2fPython-blue)
![Frontend Mobile](https://img.shields.io/badge/Frontend-React_Native_PWA-blueviolet)

This repository contains the codebase and evaluation outputs for a **Hybrid Voice Spoofing Detection System**. It explores the challenge of distinguishing between genuine human voices and AI-generated synthesised voices (voice spoofing) via microphone recordings. 

The system utilizes a hybrid approach: processing Mel-Frequency Cepstral Coefficients (MFCC) via a 2D Convolutional Neural Network (CNN), combined with rule-based heuristics that analyze spectral flux, zero-crossing rate (ZCR), and spectral flatness to penalise characteristics typical of synthetic spoof audio. The results feature a strong multi-window voting mechanism to aggregate CNN scores and provide a resilient final classification.

The project features both a **Web interface** and a mobile-optimized **Progressive Web App (PWA)** built with Expo, to simulate real-world mobile environments (e.g., handling iOS microphone capture limitations securely via HTTPS tunnels).

## 🚀 Key Features

*   **Hybrid Analysis:** Uses a trained CNN alongside spectral characteristics (Spectral Flux CV, ZCR Standard Deviation) to weed out artificially smooth voices.
*   **Multi-window Voting:** Splits long audio segments into overlapping windows (3 seconds each) avoiding singular score spikes allowing AI evasion.
*   **Active-Speech Trimming:** Filters out silence/breath sounds. Uses K-Means clustering to distinguish if multiple distinct speakers exist in an audio sequence.
*   **On-the-fly Dataset Collection:** Supports continuous learning by allowing users to classify an audio track as True Real or True Spoof from the interface and saving it directly to local datasets.
*   **React Native PWA:** Provides a clean iOS/Android mobile user experience handling recording logic locally and uploading effectively.

---

## 📈 Evaluation Results & Outputs

The model underwent rigorous testing over unseen audio samples. Below are the generated visual evaluation results.

*(All visuals are stored in `backend/training/evaluation_results/`)*

### Performance Metrics Summary

*   **Overall Accuracy:** 99.89%
*   **Per-Class Accuracy:**
    *   `Real Voice` - 100.00%
    *   `Spoofed Voice` - 99.78%
*   **ROC AUC Score:** 0.9997

### Visual Dashboards

<details>
<summary><b>1. Confusion Matrix</b> <i>(Click to expand)</i></summary>
<br/>
<img src="backend/training/evaluation_results/confusion_matrix.png" alt="Confusion Matrix" width="600"/>
</details>

<details>
<summary><b>2. Score Distribution (Confidence Map)</b> <i>(Click to expand)</i></summary>
<br/>
<img src="backend/training/evaluation_results/score_distribution.png" alt="Score Distribution Map" width="600"/>
</details>

<details>
<summary><b>3. ROC Curve</b> <i>(Click to expand)</i></summary>
<br/>
<img src="backend/training/evaluation_results/roc_curve.png" alt="Receiver Operating Characteristic Curve" width="600"/>
</details>

<details>
<summary><b>4. Precision-Recall Curve</b> <i>(Click to expand)</i></summary>
<br/>
<img src="backend/training/evaluation_results/precision_recall_curve.png" alt="Precision Recall Curve" width="600"/>
</details>

<details>
<summary><b>5. Per-Class Accuracy</b> <i>(Click to expand)</i></summary>
<br/>
<img src="backend/training/evaluation_results/per_class_accuracy.png" alt="Per Class Accuracy" width="600"/>
</details>

<details>
<summary><b>6. Metrics Summary Hex/Grid</b> <i>(Click to expand)</i></summary>
<br/>
<img src="backend/training/evaluation_results/metrics_summary.png" alt="General Metrics Review" width="600"/>
</details>


## 📂 Project Structure

```text
├── backend/                  # Flask microservice ecosystem
│   ├── app.py                # Main backend exposing /predict logic and MFCC operations
│   ├── model/                # .keras models trained 
│   ├── training/             # Scripts to independently retrain & evaluate models
│   │   └── evaluation_results/ # Graphical outputs mapped above
│   ├── start_server.bat      # Windows batch starter file
│   └── templates/            # Core backend dashboard interfaces
├── mobile-app/               # React Native PWA via Expo
│   ├── App.js                # Core layout and recording interface components
│   └── package.json          # Dependencies required for node environment
├── dataset/                  # Structured local WAV and MP3 data 
│   ├── real_mic_recorded/
│   └── spoofed_recorded/
└── poster.html               # Presentation/Research academic poster
```

## 🛠 Usage & Setup

### Running Backend (Python)
Ensure Python 3.10+ is natively installed or running in an environment. 
1. Navigate to the `backend/` directory.
2. `pip install -r requirements.txt` (to install Librosa, TensorFlow, Flask, etc.)
3. `python app.py` (or run `start_server.bat` on Windows).

By default, the server runs securely (`adhoc` SSL via Flask for recording access).

### Running Mobile Frontend (React Native PWA)
1. Navigate to the `mobile-app/` directory.
2. `npm install`
3. `npx expo start --web` (to execute locally in a web browser) or `npx expo start` to run on Expo Go for mobile devices.

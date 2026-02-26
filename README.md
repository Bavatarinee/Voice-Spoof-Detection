Features

🎙️ Live Microphone Recording – Record voice directly in the browser

📂 Audio File Upload – Supports WAV, MP3, FLAC, OGG, WebM, M4A

📊 Real-time Waveform Visualizer – Live oscilloscope-style display

🧠 Multi-Window Voting – 3-second overlapping windows (50% overlap)

🔍 Rule-Based Spoof Penalty – Detects spectral flatness, ZCR variance, spectral flux

📈 Per-Window Confidence Breakdown – Visual score bars

💾 Dataset Collection Tool – Save samples for retraining

🔒 HTTPS Enabled – Required for microphone access

Detection Pipeline
Raw Audio Input (Mic/File)
        ↓
Resample to 22050 Hz
        ↓
Amplitude Normalization
        ↓
Slice into 3s overlapping windows (50% overlap)
        ↓
Extract 40 MFCCs → (40 × 94) feature matrix
        ↓
CNN Model Inference
        ↓
Trimmed Mean Aggregation
        ↓
Rule-Based Spoof Penalty
        ↓
Majority Vote Guard (≥55% REAL)
        ↓
Final Decision:
REAL VOICE ✅  |  SPOOF VOICE 🚨


CNN Architecture
Layer	Details
Conv Block 1	32 filters, 3×3, ReLU + BN + MaxPool + Dropout(0.25)
Conv Block 2	64 filters, 3×3, ReLU + BN + MaxPool + Dropout(0.25)
Conv Block 3	128 filters, 3×3, ReLU + BN + MaxPool + Dropout(0.25)
Dense Layers	256 → 64 → 1 (Sigmoid)
Optimizer	Adam (lr = 0.0003)
Loss	Binary Crossentropy


Project Structure
voice-spoof-detection/
│
├── backend/
│   ├── app.py
│   ├── predict.py
│   ├── model/
│   │   └── voice_model.keras
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   └── training/
│       ├── train_model.py
│       └── evaluate_model.py
│
├── dataset/
│   ├── real_samples/
│   ├── real_mic_recorded/
│   ├── FlashSpeech/
│   ├── NaturalSpeech3/
│   ├── OpenAI/
│   ├── PromptTTS2/
│   ├── seedtts_files/
│   ├── VALLE/
│   ├── VoiceBox/
│   ├── xTTS/
│   └── spoofed_recorded/
│
├── requirements.txt


Train the Model
python backend/training/train_model.py


Evaluate the Model (Optional)
python backend/training/evaluate_model.py

Run the Server
python backend/app.py


Future Improvements

Transformer-based models (Wav2Vec, HuBERT)

Streaming inference via WebSockets

Cross-platform deployment (TensorFlow Lite / ONNX)

Explainable AI (Grad-CAM, SHAP)

ASVspoof 2024 dataset integration

📜 License

This project is for academic and research purposes.

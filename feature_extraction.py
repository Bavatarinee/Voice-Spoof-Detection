import librosa
import numpy as np

SAMPLE_RATE = 16000
DURATION = 3  # seconds

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    # Create Mel Spectrogram (like training)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize to fixed shape (important!)
    if mel_spec_db.shape[1] < 128:
        pad_width = 128 - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db,
                             pad_width=((0, 0), (0, pad_width)),
                             mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :128]

    return mel_spec_db
    
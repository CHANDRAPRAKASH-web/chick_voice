
import os
import torch
import numpy as np
import librosa

from model import ChickenResNet   # we already defined this in model.py


# ---------- SETTINGS (must match training) ----------
SAMPLE_RATE = 16000
DURATION = 3.0          # seconds
N_MELS = 64             # same as dataset.py
CHECKPOINT_PATH = "models/chicken_resnet_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------


def load_model_and_labels(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    label2idx = checkpoint["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}

    num_classes = len(label2idx)
    model = ChickenResNet(num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, idx2label


def preprocess_wav(path: str,
                   sample_rate: int = SAMPLE_RATE,
                   duration: float = DURATION,
                   n_mels: int = N_MELS) -> torch.Tensor:
    """Load wav -> fixed length -> mel spectrogram -> normalized tensor."""
    # load audio
    audio, sr = librosa.load(path, sr=sample_rate)

    # pad/crop to fixed length
    target_len = int(sample_rate * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # normalize same as dataset
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # to tensor [1, 1, n_mels, time]
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel_tensor


def predict_file(audio_path: str):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            f"Train first to create the file."
        )

    print(f"Loading model from: {CHECKPOINT_PATH}")
    model, idx2label = load_model_and_labels(CHECKPOINT_PATH, DEVICE)
    print("Available classes:", idx2label)

    print(f"\nPreprocessing audio: {audio_path}")
    x = preprocess_wav(audio_path)  # [1, 1, n_mels, T]
    x = x.to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = idx2label[pred_idx]
    pred_conf = float(probs[pred_idx])

    print("\n========= PREDICTION RESULT =========")
    print(f"File       : {audio_path}")
    print(f"Pred class : {pred_label}")
    print(f"Confidence : {pred_conf:.3f}")
    print("All probs  :")
    for i, p in enumerate(probs):
        print(f"  class {idx2label[i]}: {p:.3f}")
    print("=====================================\n")


if __name__ == "__main__":
    # TODO: change this to the file you want to test
    # Example: take one file from data/healthy or data/unhealthy
    AUDIO_PATH = r"data\unhealthy\79.wav"  # <-- put a real file path here

    if not os.path.exists(AUDIO_PATH):
        print("Please set AUDIO_PATH in infer.py to a real .wav file path.")
    else:
        print("Using device:", DEVICE)
        predict_file(AUDIO_PATH)
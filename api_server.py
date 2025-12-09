
# src/api_server.py

import os
import io
from typing import Dict, Optional

import numpy as np
import librosa
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model import ChickenResNet  # your ResNet-18 model


# ===== SETTINGS (match training) =====
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 64
CHECKPOINT_PATH = "models/chicken_resnet_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================


# ===== RESPONSE MODEL (JSON ONLY) =====
class PredictionResponse(BaseModel):
    pred_label: str
    status_message: str
    confidence: float               # 0–1
    confidence_percent: int         # %
    probabilities: Dict[str, float] # class -> % (0–100)
# =====================================


app = FastAPI(
    title="Chicken Voice Health API",
    description="Classifies chicken audio as Healthy / Unhealthy / Noise.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[torch.nn.Module] = None
idx2label: Optional[Dict[int, str]] = None


# ========== MODEL LOADING ==========
def load_model_and_labels():
    global model, idx2label

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            f"Train the model first to create this file."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    label2idx = checkpoint["label2idx"]
    idx2label_local = {v: k for k, v in label2idx.items()}

    num_classes = len(label2idx)
    m = ChickenResNet(num_classes=num_classes)
    m.load_state_dict(checkpoint["state_dict"])
    m.to(DEVICE)
    m.eval()

    model = m
    idx2label = idx2label_local


@app.on_event("startup")
def startup_event():
    print("Loading model on startup...")
    load_model_and_labels()
    print("Model loaded on", DEVICE)
# ==================================


# ========== PREPROCESSING ==========
def preprocess_audio_from_bytes(file_bytes: bytes):
    audio_buf = io.BytesIO(file_bytes)
    audio, sr = librosa.load(audio_buf, sr=SAMPLE_RATE)

    target_len = int(SAMPLE_RATE * DURATION)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel_tensor


def predict_from_mel(mel_tensor: torch.Tensor):
    if model is None or idx2label is None:
        raise RuntimeError("Model not loaded")

    mel_tensor = mel_tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = idx2label[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs
# ===================================


# ========== HELPERS ==========
def make_status_message(pred_label: str, confidence_percent: int) -> str:
    l = pred_label.lower()
    if l == "healthy":
        return f"Chicken is Healthy ({confidence_percent}%)"
    elif l == "unhealthy":
        return f"Chicken is Unhealthy ({confidence_percent}%)"
    else:
        return f"Noise Detected ({confidence_percent}%)"
# =============================


@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # 1) Validate file type
    if file.content_type not in ["audio/wav", "audio/x-wav", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Please upload a .wav audio file")

    # 2) Read bytes
    file_bytes = await file.read()

    # 3) Preprocess & predict
    try:
        mel_tensor = preprocess_audio_from_bytes(file_bytes)
        pred_label, confidence, probs = predict_from_mel(mel_tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

    # 4) Convert to percentages
    confidence_percent = int(round(confidence * 100))

    prob_dict: Dict[str, float] = {}
    for i, p in enumerate(probs):
        label = idx2label[i]
        prob_dict[label] = round(float(p) * 100, 2)

    status_message = make_status_message(pred_label, confidence_percent)

    # 5) Return clean JSON ONLY
    return PredictionResponse(
        pred_label=pred_label,
        status_message=status_message,
        confidence=confidence,
        confidence_percent=confidence_percent,
        probabilities=prob_dict,
    )
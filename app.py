import torch
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from model import SleepTransformer


CHECKPOINT_PATH = "best_model_4class.pt"

CLASS_NAMES = {
    0: "Wake",
    1: "Light",
    2: "Deep",
    3: "REM",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Sleep Stage Classification API")

model = None
scaler = None
config = None


class SleepWindowRequest(BaseModel):
    samples: List[List[float]]


class SleepWindowResponse(BaseModel):
    class_id: int
    stage: str
    is_light_sleep: bool
    light_sleep_probability: float
    probabilities: Dict[str, float]


@app.on_event("startup")
def load_model():
    global model, scaler, config

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    config = checkpoint["config"]
    scaler = checkpoint["scaler"]

    model = SleepTransformer(
        n_channels=len(config["feature_columns"]),
        seq_len=config["window_size"],
        n_classes=config["n_classes"],
        model_dim=config["model_dim"],
        num_heads=config["num_attention_heads"],
        num_layers=config["num_transformer_layers"],
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


@app.get("/")
def root():
    return {
        "message": "sleep stage api is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
    }


@app.post("/predict", response_model=SleepWindowResponse)
def predict_sleep_stage(request: SleepWindowRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    x = np.array(request.samples, dtype=np.float32)

    # derive from the loaded checkpoint so this stays in sync with the model (8 or 9 ch)
    expected_shape = (config["window_size"], len(config["feature_columns"]))

    if x.shape != expected_shape:
        raise HTTPException(
            status_code=400,
            detail=f"expected input shape {expected_shape}, got {x.shape}",
        )

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # same preprocessing as training
    # feature order (8 ch, EDA dropped): F4-M1, C4-M1, ACC_X, ACC_Y, ACC_Z, TEMP, HR, IBI
    x[:, 0] *= 1e6   # F4-M1 -> uV
    x[:, 1] *= 1e6   # C4-M1 -> uV

    x = scaler.transform(x)

    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    class_id = int(np.argmax(probs))
    stage = CLASS_NAMES[class_id]
    light_sleep_probability = float(probs[1])

    return {
        "class_id": class_id,
        "stage": stage,
        "is_light_sleep": light_sleep_probability >= 0.60,
        "light_sleep_probability": light_sleep_probability,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(probs))
        },
    }
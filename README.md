# Sleep Stage Classification API

A small [FastAPI](https://fastapi.tiangolo.com/) service that classifies a **30‑second
window of wearable sensor data** into a sleep stage — **Wake / Light / Deep / REM** —
using a multi‑scale CNN + Transformer model.

Live: `https://sleep-stage-api.onrender.com` · Interactive docs at `/docs`.

---

## How it works

```
POST /predict
  { "samples": [ 3000 rows × 8 channels of RAW values ] }
        │
        ▼
  shape check  →  EEG cols ×1e6 (V→µV)  →  StandardScaler  →  SleepTransformer  →  softmax
        │                                                                            │
        └──────────────────────────────► { stage, probabilities, ... } ◄────────────┘
```

- A window is **30 s at 100 Hz = 3000 timesteps**, with **8 channels** per timestep.
- The trained model **and** its scaler are loaded once at startup from
  `best_model_4class.pt`. The expected input shape is read from the checkpoint's
  `config` (`window_size` × `len(feature_columns)`), so the API automatically tracks
  the model if the channel set ever changes.
- **You send raw values.** The server does the EEG `×1e6` conversion and applies the
  `StandardScaler` from training — do **not** standardize or pre‑scale on the client.

### Model (`model.py`)

`SleepTransformer`:
1. **Multi‑scale 1D CNN** — four parallel conv branches (kernel sizes 5 / 25 / 50 / 100)
   capture features at different temporal scales, concatenated and fused.
2. **Positional encoding** + **Transformer encoder** (GELU, `batch_first`).
3. **Attention pooling ⊕ mean pooling** → concatenated.
4. **MLP classifier** → 4 logits.

Exact hyper‑parameters (`model_dim`, heads, layers, dropout) come from the checkpoint's
`config`, so the served architecture always matches the trained weights.

---

## API

### `GET /`
Liveness. `{ "message": "sleep stage api is running", "docs": "/docs" }`

### `GET /health`
`{ "status": "ok", "device": "cpu" | "cuda" }`

### `POST /predict`
Classify one window.

#### Request body

```json
{ "samples": [[f0, f1, f2, f3, f4, f5, f6, f7], "... 3000 rows total ..."] }
```

`samples` must be **exactly 3000 rows × 8 columns**, in this channel order:

| # | Channel | Sensor | **Send in unit** | Server transform |
|---|---------|--------|------------------|------------------|
| 0 | `F4-M1` | EEG, frontal (ADS1292R) | **volts (V)** | `×1e6` → µV |
| 1 | `C4-M1` | EEG, central            | **volts (V)** | `×1e6` → µV |
| 2 | `ACC_X` | Accelerometer | 1/64‑g counts | — |
| 3 | `ACC_Y` | Accelerometer | 1/64‑g counts | — |
| 4 | `ACC_Z` | Accelerometer | 1/64‑g counts | — |
| 5 | `TEMP`  | Skin temperature | °C | — |
| 6 | `HR`    | Heart rate | bpm | — |
| 7 | `IBI`   | Inter‑beat interval | **seconds** | — |

> ⚠️ **Load‑bearing gotchas** (mismatches here silently corrupt a channel and ruin the
> prediction — no error is raised):
> - **EEG in volts, not microvolts.** The server multiplies columns 0–1 by `1e6`. Raw
>   amplitudes are ~±50 µV ≈ `±5e‑5` V.
> - **IBI in seconds (~0.95), not milliseconds.** Sending ms (~950) blows the channel up
>   by 1000×.
> - **Send raw, unscaled values.** The `×1e6` and the `StandardScaler` are applied
>   server‑side.
> - `NaN`/`Inf` are replaced with `0` automatically.

#### Response

```json
{
  "class_id": 1,
  "stage": "Light",
  "is_light_sleep": true,
  "light_sleep_probability": 0.71,
  "probabilities": { "Wake": 0.10, "Light": 0.71, "Deep": 0.05, "REM": 0.14 }
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `class_id` | int | argmax class, 0–3 |
| `stage` | str | `Wake` / `Light` / `Deep` / `REM` |
| `light_sleep_probability` | float | softmax prob of the `Light` class |
| `is_light_sleep` | bool | `light_sleep_probability >= 0.60` |
| `probabilities` | object | full softmax distribution (sums to 1) |

#### Errors
- `400` — input shape ≠ `(3000, 8)` (message reports the shape you sent).
- `503` — model not loaded yet (e.g. mid cold‑start).

---

## Sleep‑stage classes

AASM stages are collapsed into 4 classes (this is how the labels were trained):

| class_id | stage | from AASM |
|----------|-------|-----------|
| 0 | Wake  | `W` |
| 1 | Light | `N1`, `N2` |
| 2 | Deep  | `N3` |
| 3 | REM   | `R` |

## Input scale sanity‑check

Approximate per‑channel **mean / std of the training distribution**, in the units the
scaler sees (so EEG is *after* the server's `×1e6`). Use these to confirm your inputs are
in the right ballpark; they are not exact and should be re‑confirmed from the checkpoint's
`scaler` if needed:

| Channel | mean / std | unit |
|---------|-----------|------|
| F4-M1 | 0.01 / 34.7 | µV |
| C4-M1 | −0.03 / 38.0 | µV |
| ACC_X | −8.6 / 35.3 | 1/64‑g |
| ACC_Y | −5.4 / 32.6 | 1/64‑g |
| ACC_Z | 25.4 / 32.6 | 1/64‑g |
| TEMP  | 33.8 / 2.43 | °C |
| HR    | 66.7 / 13.2 | bpm |
| IBI   | 0.949 / 0.241 | s |

---

## Examples

**Python**

```python
import requests

URL = "https://sleep-stage-api.onrender.com/predict"

# one 30 s window: 3000 rows, each [F4-M1, C4-M1, ACC_X, ACC_Y, ACC_Z, TEMP, HR, IBI]
window = [[0.0] * 8 for _ in range(3000)]   # replace with real samples

r = requests.post(URL, json={"samples": window})
r.raise_for_status()
print(r.json()["stage"], r.json()["light_sleep_probability"])
```

**curl**

```bash
curl -X POST https://sleep-stage-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"samples": [[0,0,0,0,0,0,0,0], "... 3000 rows of 8 floats ..."]}'
```

---

## Model checkpoint (`best_model_4class.pt`)

A `torch.save` dict, loaded with `weights_only=False`:

| key | contents |
|-----|----------|
| `model_state_dict` | trained `SleepTransformer` weights |
| `scaler` | fitted scikit‑learn `StandardScaler` |
| `config` | `feature_columns`, `window_size`, `n_classes`, `model_dim`, `num_attention_heads`, `num_transformer_layers`, `dropout` |

To deploy a new model, replace this file (keep the same dict structure) and push.

> Because `scaler` is a pickled scikit‑learn object and the checkpoint is loaded with
> `weights_only=False`, the `torch` / `scikit-learn` versions on the server must be
> import‑compatible with the ones used to train. If the scaler fails to unpickle after a
> retrain, pin those versions in `requirements.txt`.

---

## Run locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build
pip install -r requirements.txt
uvicorn app:app --reload      # → http://127.0.0.1:8000/docs
```

## Deploy (Render / Docker)

- `Dockerfile` builds on `python:3.11-slim`, installs `requirements.txt`, and runs
  `uvicorn app:app --host 0.0.0.0 --port ${PORT}`.
- The model is baked into the image (`COPY . .`), so **updating the model = commit the new
  `best_model_4class.pt`**.
- Render builds from the `main` branch; push to `main` to trigger a redeploy.
- On the free tier the service sleeps when idle — the **first request after a cold start**
  loads the model and can take ~30–60 s (`503` until ready).

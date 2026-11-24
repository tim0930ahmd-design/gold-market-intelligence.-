import os
import time
import threading
from typing import Any

import numpy as np
import torch
import timesfm
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
API_KEY = os.environ.get("TIMESFM_API_KEY", "")
HF_HOME = os.environ.get("HF_HOME", "/tmp/huggingface")
os.environ["HF_HOME"] = HF_HOME
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
torch.set_float32_matmul_precision("high")

app = FastAPI(title="TimesFM XAUUSD Assistant", version="0.1.0")
_model = None
_model_lock = threading.Lock()
_model_error = None
_model_loaded_at = None


class Candle(BaseModel):
    time: str
    close: float


class ForecastRequest(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    current_price: float = Field(gt=0)
    candles: list[Candle] = Field(min_length=64, max_length=2048)
    horizons: list[int] = Field(default_factory=lambda: [1, 3, 6])


def load_model():
    global _model, _model_error, _model_loaded_at
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            candidate = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                MODEL_ID,
                cache_dir=HF_HOME,
            )
            candidate.compile(
                timesfm.ForecastConfig(
                    max_context=512,
                    max_horizon=32,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            _model = candidate
            _model_error = None
            _model_loaded_at = time.time()
            return _model
        except Exception as exc:
            _model_error = f"{type(exc).__name__}: {exc}"[:500]
            raise


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "timesfm-xauusd-assistant",
        "model": MODEL_ID,
        "loaded": _model is not None,
        "modelError": _model_error,
        "loadedAt": _model_loaded_at,
        "mode": "FORECAST_ONLY",
        "tradeExecution": False,
    }


@app.post("/forecast")
def forecast(payload: ForecastRequest, x_api_key: str | None = Header(default=None)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")
    if payload.symbol.upper().replace("/", "") not in {"XAUUSD", "GOLD"}:
        raise HTTPException(status_code=400, detail="UNSUPPORTED_SYMBOL")
    if payload.timeframe.upper() != "M5":
        raise HTTPException(status_code=400, detail="UNSUPPORTED_TIMEFRAME")
    horizons = sorted(set(int(h) for h in payload.horizons))
    if not horizons or horizons[0] < 1 or horizons[-1] > 30:
        raise HTTPException(status_code=400, detail="HORIZON_OUT_OF_RANGE")
    closes = np.asarray([c.close for c in payload.candles], dtype=np.float32)
    if not np.isfinite(closes).all() or np.any(closes <= 0):
        raise HTTPException(status_code=400, detail="INVALID_CLOSE_SERIES")
    if abs(float(closes[-1]) - payload.current_price) > max(5.0, payload.current_price * 0.01):
        raise HTTPException(status_code=400, detail="CURRENT_PRICE_MISMATCH")
    try:
        model = load_model()
        started = time.perf_counter()
        point, quantiles = model.forecast(
            horizon=horizons[-1],
            inputs=[closes[-512:]],
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        forecasts: dict[str, Any] = {}
        for h in horizons:
            idx = h - 1
            median = float(point[0, idx])
            p10 = float(quantiles[0, idx, 1])
            p50 = float(quantiles[0, idx, 5])
            p90 = float(quantiles[0, idx, 9])
            forecasts[str(h * 5)] = {
                "horizonBars": h,
                "horizonMinutes": h * 5,
                "median": median,
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "change": median - payload.current_price,
                "changePct": (median / payload.current_price - 1.0) * 100.0,
            }
        return {
            "ok": True,
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "model": MODEL_ID,
            "modelVersion": "TimesFM-2.5-200M",
            "forecastedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inputBars": int(len(closes)),
            "lastInputTime": payload.candles[-1].time,
            "currentPrice": float(payload.current_price),
            "forecasts": forecasts,
            "latencyMs": latency_ms,
            "mode": "FORECAST_ONLY",
            "note": "Quantiles describe forecast ranges; they are not calibrated trade-win probabilities.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"TIMESFM_INFERENCE_FAILED: {type(exc).__name__}: {exc}"[:500])


@app.on_event("startup")
def warm_model():
    try:
        load_model()
        print("TIMESFM_MODEL_READY", MODEL_ID, flush=True)
    except Exception as exc:
        print("TIMESFM_MODEL_LOAD_FAILED", type(exc).__name__, str(exc)[:500], flush=True)

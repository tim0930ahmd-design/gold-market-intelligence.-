# TiRex-2 observer sidecar

This service is the isolated forecasting helper for Gold AI. It uses the upstream CPU inference image:

`ghcr.io/nx-ai/tirex2-cpu`

## Deployment contract

- Deploy as a separate service; do not replace or modify the Gemini/primary engine or cTrader Bridge.
- Container listens on port `8000`.
- Keep it on the private Railway network only. The upstream HTTP API has **no authentication** and must not be exposed publicly.
- Set `TIREX2_URL` on the primary Gold AI service to the sidecar's private HTTP origin (for example, `http://<private-service-host>:8000`).
- Keep the primary engine's existing persistent `/data` volume attached. TiRex forecasts and Gemini decision snapshots are appended to the existing state file; the new endpoint `GET /model-history` is protected by the existing `PAPER_API_TOKEN`.
- No model forecast can open, reject, or modify a PAPER candidate. TiRex is observation-only.

## Request

The primary engine sends the latest 256 closed M5 closes to:

`POST /univariate/forecast/quantiles`

with `prediction_length: 6`. It records the raw response, median forecast where recognized, model status, source bar, latency, and later actual prices at +1, +3 and +6 M5 bars.

## Verification required before production use

1. Deploy the CPU image on a private network and confirm `GET /health`.
2. Confirm a sample quantile request returns a valid response.
3. Confirm primary engine logs show `tirex2.status=AVAILABLE`.
4. Confirm `/model-history` contains forecast and Gemini snapshots and later actual outcomes.
5. Confirm existing Gemini decisions, PAPER candidate decisions, Kronos service, cTrader Bridge, and `/data` remain unchanged.

The official upstream image bundles the default model weights. Its HTTP API is unauthenticated, so private-network isolation is mandatory.

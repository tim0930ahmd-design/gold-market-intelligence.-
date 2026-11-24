# TimesFM XAUUSD Assistant

Isolated, forecast-only sidecar for the existing Gold Playbook. It does not create, approve, reject, or execute trades.

- Model: Google TimesFM 2.5, 200M PyTorch checkpoint.
- Input: closed M5 XAUUSD closes and current quote.
- Output horizons: 5, 15, and 30 minutes by default, with median and P10/P50/P90 ranges.
- Authentication: `X-API-Key` header.
- Health: `GET /health`.
- Forecast: `POST /forecast`.

The model is loaded independently from the main Bun/Gemini service. The sidecar has no access to the Playbook state volume and no broker credentials.

# Gold AI Intelligence Engine v0.6 — Advanced Learning Layers

Status: implementation on the development branch only. PAPER only. Not deployed and not approved for live trading.

## What was added

### 1) AI Opportunity Discovery
- Builds a 24-value causal feature vector from completed M5 candles: multi-horizon returns, candle geometry, efficiency, volatility expansion/compression, range position, slopes and UTC session encoding.
- Collects forward labels for BUY and SELL independently: first touch of +1.0 ATR target versus -0.75 ATR stop over the next 12 completed M5 bars. If neither barrier is reached, label TIMEOUT. If both are hit in one bar, STOP wins conservatively.
- A separately trained binary model can propose AI_DISCOVERY candidates without requiring a hand-coded setup trigger. TIMEOUT remains in training as a non-target outcome with zero gross payoff (costs still apply).
- Runtime will ignore the discovery model unless its artifact has kind `gold-opportunity-linear-v1` and `validated: true`.

### 2) Scenario Prediction
- Uses the same causal feature vector.
- Trains separate three-class probability forecasts for 3, 6 and 12 M5-bar horizons: UP, DOWN or RANGE.
- Labels use the future close return in ATR units: UP above +0.35 ATR, DOWN below -0.35 ATR, otherwise RANGE.
- Probabilities are displayed as forecasts, not used as a forced consensus gate.

### 3) Meta-Learning
- Learns smoothed expert weights from closed PAPER trades and shadow outcomes, conditioned on the current regime.
- Needs at least 12 observations for an expert/regime pair before adapting its weight; otherwise weight remains neutral at 1.0.
- Ranks opportunities by the independent expert score multiplied by the learned weight. It does not require all experts to agree.

### 4) Decision Intelligence
- Uses the existing XGBoost TARGET/STOP/TIMEOUT meta-labeler, now applying its saved Platt calibration coefficients at inference.
- Calculates expected USD value at both $0.37 base cost and $0.74 stress cost.
- PAPER entry is eligible only when the validated meta-labeler gives positive expected value under both cost assumptions and the existing risk checks pass.
- No validated model means NO_TRADE while outcomes continue to be collected. No probabilities are fabricated.

### 5) Feedback / learning loop
- Each newly completed M5 bar adds one feature snapshot.
- The engine resolves labels only after the required future bars have completed; no future data is used in the feature vector.
- Pending snapshots and resolved samples are stored in the persistent PAPER state as `aiLearningPending` and `aiLearningSamples`.
- Samples are capped at 12,000; pending rows are capped at 500.

## Training

Python dependencies are listed in `requirements.txt`.

Run against a copy of the persistent PAPER state:
```bash
python ml/train_advanced_intelligence.py --state /data/gold-ai-paper-state.json --out-dir /data --cost 0.37 --stress-cost 0.74 --lot-oz 1
python ml/train_xgboost.py --state /data/gold-ai-paper-state.json --out /data/gold-ai-xgb-model.json
```

Advanced discovery model gate:
- At least 500 labelled BUY and 500 labelled SELL observations.
- Chronological 70% train / 15% threshold selection / 15% untouched test.
- Brier score at least 2% better than the test base-rate predictor.
- Threshold selected only on the middle chronological slice.
- At least 30 untouched-test trades, PF >= 1.25 at $0.37, positive net at $0.37, and PF > 1.0 plus positive net at $0.74.
- The model artifact is not written unless both BUY and SELL pass.

Scenario model gate:
- At least 900 labelled samples per horizon and all three classes represented in train and test.
- Chronological split; Brier score at least 2% better than the untouched-test base-rate predictor for each of 3, 6 and 12 bars.
- The scenario artifact is written only if all three horizons pass.

XGBoost meta-label gate remains chronological and separate. A model is not trusted merely because training completed.

## Current truth / limitations

- No trained opportunity-discovery or scenario artifact is included in this commit.
- No new model has been trained or validated from live PAPER samples yet.
- The existing historical proxy replay was negative; it is not relabelled as success.
- The new learning loop must collect sufficient forward outcomes before artifacts can pass their gates.
- Python training is an offline step; it is not automatically run by the Bun service.
- This development branch imports `advanced_intelligence.ts`. Railway's existing start command must fetch that file as well as `index.ts` and `ai_layers.ts` before any deployment. No Railway deployment was performed.
- No Bridge, existing production engine, broker execution, or live trading changes.

## Files
- `advanced_intelligence.ts`: feature extraction, discovery/scenario inference, expert learning, EV decision, forward-only labels.
- `index.ts`: PAPER runtime integration and status reporting.
- `ai_layers.ts`: regime routing, adaptive expert selection and calibrated XGBoost inference.
- `ml/train_advanced_intelligence.py`: discovery/scenario model training and validation.
- `ml/train_xgboost.py`: XGBoost meta-label training.

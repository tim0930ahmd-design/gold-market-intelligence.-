# Gold AI v0.5 — Research Gate (not a trading approval)

## Scope
This is an exploratory historical replay of the eight specialist trigger families using the stored XAU/USD M1 market-state snapshots and intelligence features:
- 2025: 353,951 M1 rows
- September 2026: 28,947 M1 rows (held out from training)

Signals were generated only from information available at each completed minute. Entries were modeled at the next M1 open. The replay used a fixed 1.0R stop-distance proxy (bounded to $0.45–$1.50), 1.8R target, 30-minute maximum horizon, conservative stop-first handling when both barriers touched in one candle, and no overlapping positions for the sequential portfolio summary. Costs were charged once per trade: $0.37 base and $0.74 stress. No separate spread charge was added because the historical snapshots do not contain a reliable bid/ask spread series.

## Candidate replay results
The 2025 training-period candidate ledger contained 30,283 opportunities; the September 2026 held-out ledger contained 2,497.

| Sample | Trades | Win rate | PF @ $0.37 | Net @ $0.37 | PF @ $0.74 | Net @ $0.74 |
|---|---:|---:|---:|---:|---:|---:|
| 2025, all candidates (overlapping diagnostic) | 30,283 | 35.74% | 0.588 | -$11,315.85 | 0.351 | -$22,520.56 |
| 2025, sequential one-position portfolio | 22,335 | 35.58% | 0.586 | -$8,505.09 | 0.352 | -$16,769.04 |
| Sep 2026, all candidates (overlapping diagnostic) | 2,497 | 33.56% | 0.612 | -$1,138.34 | 0.419 | -$2,062.23 |
| Sep 2026, sequential one-position portfolio | 1,924 | 32.90% | 0.593 | -$934.09 | 0.407 | -$1,645.97 |

The BUY/SELL split on the held-out September sample was also negative:
- BUY: 1,201 candidates, 31.56% wins, PF 0.563 at $0.37; PF 0.386 at $0.74.
- SELL: 1,296 candidates, 35.42% wins, PF 0.661 at $0.37; PF 0.451 at $0.74.

## XGBoost meta-label experiment
Three one-vs-rest XGBoost classifiers (TARGET, STOP, TIMEOUT) were trained on the first 80% of 2025 candidates and calibrated on the final 20% of 2025 only. September 2026 was kept separate as external OOS.

| Class | 2025 chronological validation AUC | 2025 validation Brier | Base-rate Brier | Sep 2026 OOS AUC |
|---|---:|---:|---:|---:|
| TARGET | 0.507 | 0.226135 | 0.226196 | 0.519 |
| STOP | 0.505 | 0.226490 | 0.226552 | 0.523 |
| TIMEOUT | 0.533 | 0.001154 | 0.001154 | 0.679* |

*September contained only two TIMEOUT labels; that AUC is not decision-grade.

Validation Brier scores improved by only a negligible amount over the base-rate predictor (far below the 2% improvement gate). On the held-out September sample, Brier scores were marginally worse than the base-rate predictor. The calibrated model produced zero positive-EV candidates out of 2,497 at the $0.37 base cost.

## Decision
**REJECT as a tradable edge. Do not deploy a trained model from this experiment.** The trigger/label setup and current feature set do not produce a useful predictive edge. This does not prove that all market-intelligence architectures fail; it means this specific first-pass expert/label construction failed the validation gate.

## Important limitations
1. The historical trigger reconstruction is a proxy replay, not a byte-for-byte replay of the live PAPER service.
2. Historical files have OHLC and derived market-state features but no dependable historical bid/ask spread, so spread was not double-counted.
3. The live service's XGBoost model remains untrained until its own PAPER candidate/outcome ledger has enough labels. The training script refuses to publish a model unless chronological validation beats the base-rate Brier score by at least 2%.
4. The current regime classifier is a deterministic quantitative classifier, not a separately trained regime ML model.
5. No live trading, broker orders, or changes to the existing Bridge/current engine were made.

This report is a research rejection, not a performance claim for the production service.

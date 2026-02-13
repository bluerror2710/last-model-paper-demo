# Live Safety Rules (Mandatory)

Use these before real-money trading.

## Risk limits
- Max open trades: 1
- Risk per trade: <= 1% equity
- Daily max loss: 2%
- Weekly max drawdown: 6%
- If hit any limit: stop bot for 24h

## Operational checks
- Exchange API healthy
- No repeated order failures
- Clock/time sync OK
- Network stable

## Go-live gate
Only go live if ALL true:
1. 2-4 weeks paper trading completed
2. Profit factor > 1.10
3. Max drawdown within your tolerance
4. No critical runtime errors

## Kill switch
Stop bot immediately if:
- 3 consecutive stop-loss events in short window
- abnormal slippage spikes
- exchange instability

Command:
```bash
pkill -f "freqtrade trade --userdir freqtrade --config freqtrade/config.paper.json"
```

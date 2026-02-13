# Last Trading Model (Paper)

Includes:
- `freqtrade/strategies/PaperSafeStrategy.py`
- `freqtrade/config.paper.json`
- `notebook_demo.ipynb`

## Quick run
```bash
source .venv/bin/activate
freqtrade backtesting --enable-protections --userdir freqtrade --config freqtrade/config.paper.json --strategy PaperSafeStrategy --timerange 20250301-20260201 --timeframe 1h
```

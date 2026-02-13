# Last Trading Model (Paper)

Includes:
- `freqtrade/strategies/PaperSafeStrategy.py`
- `freqtrade/config.paper.json`
- `notebook_demo.ipynb`
- `streamlit_app.py`
- `LIVE_SAFETY_RULES.md`

## Quick run
```bash
source .venv/bin/activate
freqtrade backtesting --enable-protections --userdir freqtrade --config freqtrade/config.paper.json --strategy PaperSafeStrategy --timerange 20250301-20260201 --timeframe 1h
```

## Streamlit app (interactive)
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud
- Repo root contains:
  - `streamlit_app.py` (entrypoint)
  - `requirements.txt`
  - `.streamlit/config.toml`
- In Streamlit Cloud, set app file to: `streamlit_app.py`

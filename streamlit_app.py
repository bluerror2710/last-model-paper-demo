import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading, time, json
from pathlib import Path

st.set_page_config(page_title="Auto Signal Pro", layout="wide")
st.title("🤖 Auto Buy / Hold / Sell — Pro Dashboard")
st.caption("Auto-tuned thresholds + evidence panels. Educational use only. Supports 1m to 1d intervals.")

with st.sidebar:
    ticker = st.selectbox("Asset", ["BTC-EUR", "ETH-EUR", "SOL-EUR", "XRP-EUR", "ADA-EUR", "AAPL", "TSLA", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "SPY", "QQQ", "GLD", "EURUSD=X", "EURJPY=X"], index=0)
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=4)


def _status_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("/", "_").replace("=", "_").replace("-", "_")
    return Path(__file__).with_name(f"bot_status_{safe}_{interval}.json")

STATUS_PATH = _status_path(ticker, interval)

def _bot_loop(symbol: str, period: str, interval: str, status_path: Path, start_capital: float = 10000.0):
    capital = start_capital
    pos = 0
    entry = 0.0
    trades = []
    while True:
        try:
            d = add_features(load_data(symbol, period, interval), interval)
            sh, rb, rs, vm, sig, sret, eq, dd = auto_tune(d)
            d["signal"] = sig
            last = d.iloc[-1]
            s_now = int(last["signal"])
            price = float(last["close"])

            if pos != 0 and s_now == -pos:
                pnl = (price - entry) / entry * pos * (capital * 0.01 * 5)
                capital += pnl
                trades.append(pnl)
                pos = 0
            if pos == 0 and s_now != 0:
                pos = s_now
                entry = price

            cum_pnl = capital - start_capital
            cum_pct = (capital / start_capital - 1) * 100
            status = {
                "ts": str(d.index[-1]),
                "symbol": symbol,
                "capital": round(capital, 2),
                "cum_pnl": round(cum_pnl, 2),
                "cum_pct": round(cum_pct, 2)
            }
            status_path.write_text(json.dumps(status, indent=2))
        except Exception as e:
            status_path.write_text(json.dumps({"error": str(e), "symbol": symbol, "interval": interval}, indent=2))

        sleep_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
        sleep_sec = sleep_map.get(interval, 300)
        time.sleep(sleep_sec)

def start_embedded_bot(symbol: str, period: str, interval: str):
    key = f"{symbol}|{period}|{interval}"
    if st.session_state.get("bot_key") != key:
        st.session_state["bot_key"] = key
        st.session_state["bot_status_path"] = str(_status_path(symbol, interval))
        t = threading.Thread(target=_bot_loop, args=(symbol, period, interval, _status_path(symbol, interval)), daemon=True)
        t.start()
    return True



# adjust period for Yahoo intraday limits
if interval == "1m" and period in ["2y", "5y", "1y"]:
    period = "7d"
elif interval in ["5m", "15m", "30m"] and period in ["2y", "5y"]:
    period = "60d"
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.rename(columns=str.lower).dropna()


def add_features(df, interval):
    c = df["close"]
    df["ema_fast"] = c.ewm(span=15, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=81, adjust=False).mean()

    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ret"] = c.pct_change()
    win = 24 if interval in ["1h", "4h"] else (60 if interval in ["1m", "5m", "15m", "30m"] else 14)
    df["vol"] = df["ret"].rolling(win).std()
    df["trend_spread"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]

    return df.dropna()


def score(df, rsi_buy, rsi_sell, vol_min):
    long_sig = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > rsi_buy) & (df["vol"] > vol_min)
    short_sig = (df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < rsi_sell) & (df["vol"] > vol_min)

    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1

    pos = sig.replace(0, np.nan).ffill().fillna(0)
    sret = pos.shift(1).fillna(0) * df["ret"].fillna(0)
    equity = (1 + sret).cumprod()
    dd = equity / equity.cummax() - 1

    if sret.std() == 0:
        sharpe = -999
    else:
        sharpe = float((sret.mean() / sret.std()) * np.sqrt(252))

    return sharpe, sig, sret, equity, dd


def auto_tune(df):
    best = (-999, 65, 35, 0.003, None, None, None, None)
    for rb in [55, 60, 65, 70]:
        for rs in [30, 35, 40, 45]:
            if rs >= rb:
                continue
            for vm in [0.001, 0.002, 0.003, 0.004, 0.006]:
                sh, sig, sret, eq, dd = score(df, rb, rs, vm)
                if sh > best[0]:
                    best = (sh, rb, rs, vm, sig, sret, eq, dd)
    return best


start_embedded_bot(ticker, period, interval)
STATUS_PATH = Path(st.session_state.get("bot_status_path", str(_status_path(ticker, interval))))

# bot progress panel (cumulative P/L only)
st.sidebar.subheader("Bot Progress")
if STATUS_PATH.exists():
    try:
        bs = json.loads(STATUS_PATH.read_text())
        cum_pnl = float(bs.get("cum_pnl", 0.0))
        cum_pct = float(bs.get("cum_pct", 0.0))
        capital = float(bs.get("capital", 10000.0))

        st.sidebar.metric("Cumulative P/L", f"{cum_pnl:+.2f} EUR", f"{cum_pct:+.2f}%")
        st.sidebar.metric("Paper Capital", f"{capital:.2f} EUR")
        st.sidebar.caption(f"Symbol/Interval: {ticker} / {interval}")

        # Progress bar centered at 50% => 0% P/L, clipped at -20%..+20%
        lo, hi = -20.0, 20.0
        clipped = max(lo, min(hi, cum_pct))
        progress = int(((clipped - lo) / (hi - lo)) * 100)
        st.sidebar.progress(progress, text=f"P/L gauge ({lo:.0f}% to +{hi:.0f}%): {cum_pct:+.2f}%")

        st.sidebar.caption("Bot runs inside Streamlit process (no separate runner needed).")
    except Exception as e:
        st.sidebar.warning(f"Bot status unavailable: {e}")
else:
    st.sidebar.info("Bot is starting… status will appear soon.")

raw = load_data(ticker, period, interval)
if raw is None or raw.empty:
    st.error("No data returned for this asset/interval. Try a larger period or different interval.")
    st.stop()

df = add_features(raw, interval)
if df is None or df.empty:
    st.error("Not enough data after feature engineering. Try another interval/period.")
    st.stop()

sh, rb, rs, vm, sig, sret, equity, dd = auto_tune(df)
if sig is None or len(sig) == 0:
    st.error("Auto-tuning failed due to insufficient data.")
    st.stop()

# Ensemble (trend + mean-reversion + breakout)
trend_sig = sig.copy()
mr_sig = pd.Series(0, index=df.index)
mr_sig[(df["rsi"] < 30) & (df["ema_fast"] > df["ema_slow"])] = 1
mr_sig[(df["rsi"] > 70) & (df["ema_fast"] < df["ema_slow"])] = -1
roll_hi = df["close"].rolling(20).max().shift(1)
roll_lo = df["close"].rolling(20).min().shift(1)
br_sig = pd.Series(0, index=df.index)
br_sig[df["close"] > roll_hi] = 1
br_sig[df["close"] < roll_lo] = -1

vote = trend_sig + mr_sig + br_sig
ens_sig = pd.Series(0, index=df.index)
ens_sig[vote >= 2] = 1
ens_sig[vote <= -2] = -1

# fallback to trend signal when ensemble has no vote
ens_sig = ens_sig.where(ens_sig != 0, trend_sig)

# risk filter: ignore weak volatility regimes
ens_sig = ens_sig.where(df["vol"] > max(vm*0.8, 0.001), 0)

# compute returns from ensemble
pos = ens_sig.replace(0, np.nan).ffill().fillna(0)
sret = pos.shift(1).fillna(0) * df["ret"].fillna(0)
equity = (1 + sret).cumprod()
dd = equity / equity.cummax() - 1

df["signal"] = ens_sig
df["strategy_ret"] = sret
df["equity"] = equity
df["drawdown"] = dd


if df.empty:
    st.error("Dataset is empty after processing.")
    st.stop()

latest = df.iloc[-1]
sig_now = int(latest["signal"])
if sig_now == 1:
    decision, color = "BUY", "green"
elif sig_now == -1:
    decision, color = "SELL", "red"
else:
    decision, color = "HOLD", "orange"

# reason breakdown
reasons = {
    "Trend": "UP" if latest["ema_fast"] > latest["ema_slow"] else "DOWN",
    "RSI": f"{latest['rsi']:.1f} (buy>{rb} / sell<{rs})",
    "Volatility": f"{latest['vol']:.4f} (min>{vm:.3f})",
}

st.markdown(f"## Signal now: :{color}[**{decision}**]")
st.caption(f"Ensemble model active | Auto params → RSI buy>{rb}, RSI sell<{rs}, Vol min>{vm:.3f} | Base score={sh:.2f}")

if STATUS_PATH.exists():
    try:
        bs = json.loads(STATUS_PATH.read_text())
        st.info(f"Bot cumulative P/L: {float(bs.get('cum_pnl',0.0)):+.2f} EUR ({float(bs.get('cum_pct',0.0)):+.2f}%)")
    except Exception:
        pass

c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"{latest['close']:.2f}")
c2.metric("Proxy return", f"{(df['equity'].iloc[-1]-1)*100:.2f}%")
c3.metric("Max drawdown", f"{df['drawdown'].min()*100:.2f}%")
c4.metric("Signal activity", f"{(df["signal"]!=0).mean()*100:.1f}%")

with st.expander("Why this signal?"):
    st.write(reasons)

# hard risk line for paper safety
if df["drawdown"].min() < -0.06:
    st.warning("Risk guard: drawdown exceeded 6% in this backtest slice.")


# Fixed reliability test:
# Tune on history BEFORE last month, test on last month (no knobs)
if len(df) > 200:
    split_idx = max(1, len(df) - max(30, len(df)//12))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    sh_t, rb_t, rs_t, vm_t, *_ = auto_tune(train_df)
    _, sig_test, sret_test, eq_test, dd_test = score(test_df.copy(), rb_t, rs_t, vm_t)
    test_df['sig_test'] = sig_test
    test_df['eq_test'] = eq_test
    test_df['dd_test'] = dd_test

    m = test_df['sig_test'] != 0
    if m.any():
        hit = (((test_df['sig_test'].shift(1) * test_df['ret']) > 0) & m).sum() / m.sum() * 100
    else:
        hit = 0.0
    rel_return = (test_df['eq_test'].iloc[-1] - 1) * 100
    rel_dd = test_df['dd_test'].min() * 100
else:
    rb_t, rs_t, vm_t = rb, rs, vm
    hit, rel_return, rel_dd = 0.0, 0.0, 0.0

st.subheader("Reliability check (fixed, no settings)")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Train window", "From ~12 months ago")
r2.metric("Test window", "Last month")
r3.metric("Signal hit rate", f"{hit:.1f}%")
r4.metric("Test return / DD", f"{rel_return:+.2f}% / {rel_dd:.2f}%")
st.caption(f"Parameters for this reliability test: RSI buy>{rb_t}, RSI sell<{rs_t}, Vol min>{vm_t:.3f}")

# simple paper bot simulation (no real money)
start_cap = st.sidebar.number_input("Paper start capital (EUR)", min_value=100.0, value=10000.0, step=100.0)
risk_per_trade = st.sidebar.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1) / 100.0

pos = 0
entry = 0.0
capital = start_cap
curve = []
trades = []
for i in range(1, len(df)):
    px = float(df["close"].iloc[i])
    sig_i = int(df["signal"].iloc[i])
    # close/open logic
    if pos != 0 and sig_i == -pos:
        pnl = (px - entry) / entry * pos * (capital * risk_per_trade * 5)
        capital += pnl
        trades.append(pnl)
        pos = 0
    if pos == 0 and sig_i != 0:
        pos = sig_i
        entry = px
    curve.append(capital)

if len(curve) == 0:
    curve = [start_cap]
curve = pd.Series(curve, index=df.index[1:1+len(curve)])
paper_dd = curve / curve.cummax() - 1
paper_ret = (curve.iloc[-1] / start_cap - 1) * 100
win_rate = (np.array(trades) > 0).mean() * 100 if trades else 0.0

pc1,pc2,pc3,pc4 = st.columns(4)
pc1.metric("Paper bot return", f"{paper_ret:.2f}%")
pc2.metric("Paper bot max DD", f"{paper_dd.min()*100:.2f}%")
pc3.metric("Paper bot win rate", f"{win_rate:.1f}%")
pc4.metric("End capital", f"{capital:.2f} EUR")
st.caption(f"Start capital: {start_cap:.2f} EUR → End capital: {capital:.2f} EUR (P/L: {capital-start_cap:+.2f} EUR)")

# tabs
T1, T2, T3, T4, T5 = st.tabs(["Price & Signals", "Performance", "Risk", "Distributions", "Live Safety"])

with T1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], name="EMA15", line=dict(width=1.1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], name="EMA81", line=dict(width=1.1)), row=1, col=1)
    bi = df.index[df["signal"] == 1]
    si = df.index[df["signal"] == -1]
    fig.add_trace(go.Scatter(x=bi, y=df.loc[bi, "close"], mode="markers", marker=dict(symbol="triangle-up", size=8), name="Buy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=si, y=df.loc[si, "close"], mode="markers", marker=dict(symbol="triangle-down", size=8), name="Sell"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(width=1.1)), row=2, col=1)
    fig.add_hline(y=rb, line_dash="dot", row=2, col=1)
    fig.add_hline(y=rs, line_dash="dot", row=2, col=1)
    fig.update_layout(height=820, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with T2:
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5,0.25,0.25])
    fig2.add_trace(go.Scatter(x=df.index, y=df["equity"], name="Equity"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=(1 + df["ret"].fillna(0)).cumprod(), name="Buy&Hold"), row=1, col=1)
    fig2.add_trace(go.Bar(x=df.index, y=df["strategy_ret"], name="Strategy returns"), row=2, col=1)
    fig2.add_trace(go.Scatter(x=curve.index, y=curve.values, name="Paper bot equity"), row=3, col=1)
    fig2.update_layout(height=700)
    st.plotly_chart(fig2, use_container_width=True)

with T3:
    rolling_sharpe = (df["strategy_ret"].rolling(80).mean() / df["strategy_ret"].rolling(80).std()) * np.sqrt(252)
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig3.add_trace(go.Scatter(x=df.index, y=df["drawdown"], name="Drawdown"), row=1, col=1)
    fig3.add_trace(go.Scatter(x=df.index, y=rolling_sharpe, name="Rolling Sharpe(80)"), row=2, col=1)
    fig3.update_layout(height=650)
    st.plotly_chart(fig3, use_container_width=True)

with T4:
    hist = np.histogram(df["strategy_ret"].dropna(), bins=60)
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=hist[1][:-1], y=hist[0], name="Return histogram"))
    fig4.update_layout(height=360, title="Strategy return distribution")
    st.plotly_chart(fig4, use_container_width=True)

    by_signal = df.groupby("signal")["ret"].mean().rename({-1: "sell", 0: "hold", 1: "buy"})

    if len(df) > 200:
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Scatter(x=test_df.index, y=test_df['eq_test'], name='Last-month equity (out-of-sample)'))
        fig_rel.update_layout(height=300, title='Reliability test equity (trained on prior data, tested on last month)')
        st.plotly_chart(fig_rel, use_container_width=True)

    st.write("Average next-period return by signal:")
    st.dataframe(by_signal.to_frame("avg_return"))

with T5:
    st.subheader("Live safety checklist")
    st.markdown("""
- Max open trades: **1**
- Risk per trade: **<=1% equity**
- Daily max loss: **2%**
- Weekly max drawdown stop: **6%**
- If any risk limit is hit: **STOP bot 24h**

Go live only after **2-4 weeks paper trading** with stable metrics.
""")
    st.code('pkill -f "freqtrade trade --userdir freqtrade --config freqtrade/config.paper.json"', language='bash')

st.markdown("### Quick interpretation")
st.markdown("- **BUY**: trend up + RSI strong + volatility active.\n- **SELL**: trend down + RSI weak + volatility active.\n- **HOLD**: conditions mixed or weak edge.")

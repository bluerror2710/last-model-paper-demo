import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Signal Lab", layout="wide")
st.title("📈 Signal Lab (Paper-Use)")
st.caption("Educational signals only. Not financial advice.")

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Asset", ["BTC-USD", "AAPL", "TSLA", "ETH-USD", "MSFT", "NVDA"], index=0)
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=0)

    st.subheader("Signal thresholds")
    rsi_buy = st.slider("RSI buy >", 40, 80, 65)
    rsi_sell = st.slider("RSI sell <", 20, 60, 35)
    vol_min = st.slider("Volatility min", 0.0, 0.05, 0.003, step=0.001)


def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.lower).dropna()
    return df


def add_indicators(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    close = df["close"]
    df["ema_fast"] = close.ewm(span=15, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=81, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    ret = close.pct_change()
    vol_win = 24 if interval in ["1h", "4h"] else 14
    df["vol"] = ret.rolling(vol_win).std()

    # 4h trend proxy
    if interval == "1h":
        df4 = df["close"].resample("4h").last().dropna().to_frame("close")
        df4["ema50_4h"] = df4["close"].ewm(span=50, adjust=False).mean()
        df4["ema200_4h"] = df4["close"].ewm(span=200, adjust=False).mean()
        up = (df4["ema50_4h"] > df4["ema200_4h"]).reindex(df.index, method="ffill")
        df["uptrend4h"] = up.astype(int)
    else:
        df["uptrend4h"] = (df["ema_fast"] > df["ema_slow"]).astype(int)

    return df.dropna()


def add_signals(df: pd.DataFrame, rsi_buy: int, rsi_sell: int, vol_min: float) -> pd.DataFrame:
    long_sig = (
        (df["ema_fast"] > df["ema_slow"]) &
        (df["rsi"] > rsi_buy) &
        (df["vol"] > vol_min) &
        (df["uptrend4h"] == 1)
    )
    short_sig = (
        (df["ema_fast"] < df["ema_slow"]) &
        (df["rsi"] < rsi_sell) &
        (df["vol"] > vol_min) &
        (df["uptrend4h"] == 0)
    )

    df["signal"] = 0
    df.loc[long_sig, "signal"] = 1
    df.loc[short_sig, "signal"] = -1

    pos = df["signal"].replace(0, np.nan).ffill().fillna(0)
    ret = df["close"].pct_change().fillna(0)
    df["strategy_ret"] = pos.shift(1).fillna(0) * ret
    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
    return df


try:
    data = load_data(ticker, period, interval)
    data = add_indicators(data, interval)
    data = add_signals(data, rsi_buy, rsi_sell, vol_min)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

latest = data.iloc[-1]
if latest["signal"] == 1:
    rec = "🟢 BUY setup"
elif latest["signal"] == -1:
    rec = "🔴 SELL/SHORT setup"
else:
    rec = "🟡 HOLD / no clear setup"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Asset", ticker)
c2.metric("Last price", f"{latest['close']:.2f}")
c3.metric("Signal now", rec)
c4.metric("Proxy return", f"{(data['equity'].iloc[-1]-1)*100:.2f}%")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.62, 0.18, 0.20])
fig.add_trace(go.Candlestick(x=data.index, open=data["open"], high=data["high"], low=data["low"], close=data["close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["ema_fast"], name="EMA15", line=dict(width=1.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["ema_slow"], name="EMA81", line=dict(width=1.2)), row=1, col=1)

buy_idx = data.index[data["signal"] == 1]
sell_idx = data.index[data["signal"] == -1]
fig.add_trace(go.Scatter(x=buy_idx, y=data.loc[buy_idx, "close"], mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=9)), row=1, col=1)
fig.add_trace(go.Scatter(x=sell_idx, y=data.loc[sell_idx, "close"], mode="markers", name="Sell", marker=dict(symbol="triangle-down", size=9)), row=1, col=1)

fig.add_trace(go.Scatter(x=data.index, y=data["rsi"], name="RSI", line=dict(width=1.2)), row=2, col=1)
fig.add_hline(y=rsi_buy, line_dash="dot", row=2, col=1)
fig.add_hline(y=rsi_sell, line_dash="dot", row=2, col=1)

fig.add_trace(go.Scatter(x=data.index, y=data["equity"], name="Equity", line=dict(width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data["drawdown"], name="Drawdown", line=dict(width=1.0)), row=3, col=1)

fig.update_layout(height=900, xaxis_rangeslider_visible=False, title=f"{ticker} — Interactive signal dashboard")
st.plotly_chart(fig, use_container_width=True)

st.subheader("How to use")
st.markdown(
    """
1. Pick asset/timeframe in sidebar.
2. Tune RSI/volatility thresholds.
3. Follow **Signal now** + markers.
4. Validate with paper trading before any real money.

**Rule of thumb:**
- Buy when green setup appears with rising EMAs.
- Sell/short when red setup appears with falling EMAs.
- Skip when yellow (no edge).
"""
)

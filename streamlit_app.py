import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Auto Signal", layout="wide")
st.title("🤖 Auto Buy / Hold / Sell")
st.caption("Thresholds are auto-found from recent data.")

with st.sidebar:
    ticker = st.selectbox("Asset", ["BTC-USD", "AAPL", "TSLA", "ETH-USD", "MSFT", "NVDA"], index=0)
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=0)


def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.rename(columns=str.lower).dropna()


def add_features(df):
    c = df['close']
    df['ema_fast'] = c.ewm(span=15, adjust=False).mean()
    df['ema_slow'] = c.ewm(span=81, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['ret'] = c.pct_change()
    win = 24 if interval in ['1h', '4h'] else 14
    df['vol'] = df['ret'].rolling(win).std()
    return df.dropna()


def score(df, rsi_buy, rsi_sell, vol_min):
    long_sig = (df['ema_fast'] > df['ema_slow']) & (df['rsi'] > rsi_buy) & (df['vol'] > vol_min)
    short_sig = (df['ema_fast'] < df['ema_slow']) & (df['rsi'] < rsi_sell) & (df['vol'] > vol_min)
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    pos = sig.replace(0, np.nan).ffill().fillna(0)
    sret = pos.shift(1).fillna(0) * df['ret'].fillna(0)
    if sret.std() == 0:
        return -999, sig, sret
    sharpe = (sret.mean() / sret.std()) * np.sqrt(252)
    return float(sharpe), sig, sret


def auto_tune(df):
    best = (-999, 65, 35, 0.003, None, None)
    for rb in [55, 60, 65, 70]:
        for rs in [30, 35, 40, 45]:
            if rs >= rb:
                continue
            for vm in [0.001, 0.002, 0.003, 0.004, 0.006]:
                sc, sig, sret = score(df, rb, rs, vm)
                if sc > best[0]:
                    best = (sc, rb, rs, vm, sig, sret)
    return best


df = add_features(load_data(ticker, period, interval))
sh, rb, rs, vm, sig, sret = auto_tune(df)
df['signal'] = sig
df['equity'] = (1 + sret).cumprod()

now = int(df['signal'].iloc[-1])
label = 'HOLD'
color = 'orange'
if now == 1:
    label = 'BUY'
    color = 'green'
elif now == -1:
    label = 'SELL'
    color = 'red'

st.markdown(f"## Signal now: :{color}[**{label}**]")
st.caption(f"Auto params → RSI buy>{rb}, RSI sell<{rs}, Vol min>{vm:.3f} | score={sh:.2f}")

c1,c2,c3 = st.columns(3)
c1.metric('Price', f"{df['close'].iloc[-1]:.2f}")
c2.metric('Proxy return', f"{(df['equity'].iloc[-1]-1)*100:.2f}%")
c3.metric('Last RSI', f"{df['rsi'].iloc[-1]:.1f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close'))
fig.add_trace(go.Scatter(x=df.index, y=df['ema_fast'], name='EMA15'))
fig.add_trace(go.Scatter(x=df.index, y=df['ema_slow'], name='EMA81'))
bi = df.index[df['signal']==1]
si = df.index[df['signal']==-1]
fig.add_trace(go.Scatter(x=bi, y=df.loc[bi,'close'], mode='markers', marker=dict(symbol='triangle-up', size=9), name='Buy pts'))
fig.add_trace(go.Scatter(x=si, y=df.loc[si,'close'], mode='markers', marker=dict(symbol='triangle-down', size=9), name='Sell pts'))
fig.update_layout(height=520, title=f'{ticker} Auto Signals')
st.plotly_chart(fig, use_container_width=True)

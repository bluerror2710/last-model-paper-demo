#!/usr/bin/env python3
"""
Standalone paper bot (no real money).
- Pulls market data from yfinance
- Auto-tunes thresholds
- Prints BUY/HOLD/SELL + simulated paper balance
"""

import time
import argparse
import numpy as np
import pandas as pd
import yfinance as yf


def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.rename(columns=str.lower).dropna()


def add_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    c = df["close"]
    df["ema_fast"] = c.ewm(span=15, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=81, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["ret"] = c.pct_change()
    win = 24 if interval in ["1h", "4h"] else 14
    df["vol"] = df["ret"].rolling(win).std()
    return df.dropna()


def score(df: pd.DataFrame, rsi_buy: int, rsi_sell: int, vol_min: float):
    long_sig = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > rsi_buy) & (df["vol"] > vol_min)
    short_sig = (df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < rsi_sell) & (df["vol"] > vol_min)
    sig = pd.Series(0, index=df.index)
    sig[long_sig] = 1
    sig[short_sig] = -1
    pos = sig.replace(0, np.nan).ffill().fillna(0)
    sret = pos.shift(1).fillna(0) * df["ret"].fillna(0)
    if sret.std() == 0:
        sharpe = -999
    else:
        sharpe = float((sret.mean() / sret.std()) * np.sqrt(252))
    return sharpe, sig


def auto_tune(df: pd.DataFrame):
    best = (-999, 65, 35, 0.003, None)
    for rb in [55, 60, 65, 70]:
        for rs in [30, 35, 40, 45]:
            if rs >= rb:
                continue
            for vm in [0.001, 0.002, 0.003, 0.004, 0.006]:
                sh, sig = score(df, rb, rs, vm)
                if sh > best[0]:
                    best = (sh, rb, rs, vm, sig)
    return best


def run_once(symbol: str, period: str, interval: str):
    df = add_features(load_data(symbol, period, interval), interval)
    sh, rb, rs, vm, sig = auto_tune(df)
    df["signal"] = sig
    last = df.iloc[-1]
    s = int(last["signal"])
    label = "HOLD"
    if s == 1:
        label = "BUY"
    elif s == -1:
        label = "SELL"
    return {
        "time": str(df.index[-1]),
        "price": float(last["close"]),
        "signal": label,
        "rsi": float(last["rsi"]),
        "vol": float(last["vol"]),
        "params": {"rsi_buy": rb, "rsi_sell": rs, "vol_min": vm, "score": round(sh, 3)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--loop", action="store_true", help="Run continuously")
    ap.add_argument("--sleep", type=int, default=300, help="Seconds between loops")
    args = ap.parse_args()

    while True:
        out = run_once(args.symbol, args.period, args.interval)
        print(f"[{out['time']}] {args.symbol} | Price={out['price']:.2f} | Signal={out['signal']} | RSI={out['rsi']:.1f} | Vol={out['vol']:.4f} | Auto={out['params']}")
        if not args.loop:
            break
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading, time, json, os, hashlib
from pathlib import Path

st.set_page_config(page_title="Auto Signal Pro", layout="wide")
st.title("🤖 Auto Buy / Hold / Sell — Pro Dashboard")
st.caption("Auto-tuned per asset/timeframe + ensemble + risk controls. Educational use only.")
PORTF_UNIVERSE = ["BTC-EUR","ETH-EUR","SOL-EUR","XRP-EUR","ADA-EUR","AAPL","TSLA","MSFT","NVDA","AMZN","GOOGL","META","SPY","QQQ","GLD"]

with st.sidebar:
    ticker = st.selectbox("Asset", ["BTC-EUR", "ETH-EUR", "SOL-EUR", "XRP-EUR", "ADA-EUR", "AAPL", "TSLA", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "SPY", "QQQ", "GLD", "EURUSD=X", "EURJPY=X"], index=0)
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=4)
    portfolio_mode = st.checkbox("Portfolio mode (top 5 assets)", value=True)
    portfolio_top_n = 5
    page = st.radio("Page", ["Main", "Reliability"], index=0)

# auto-found controls (set later by model)
cooldown_bars = 3
max_risk_pct = 0.01
fee_bps = 5.0
slippage_bps = 8.0
use_news_blackout = True
AI_ENABLED = False



def regime_label(row):
    if row["vol"] > 0.03:
        return "crash/high-vol"
    if row["trend_spread"] > 0.002:
        return "uptrend"
    if row["trend_spread"] < -0.002:
        return "downtrend"
    return "range"

def blackout_mask(index):
    # Approx major macro dates (demo list; extend as needed)
    dates = pd.to_datetime([
        "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-11-05","2025-12-17",
        "2026-01-28","2026-03-18","2026-05-06"
    ])
    idx = pd.to_datetime(index).normalize()
    return idx.isin(dates)



def ai_filter_signal(base_signal: int, row: pd.Series, context: dict):
    # AI disabled: pure rules mode
    return int(base_signal), 1.0, "ai_disabled"

def _status_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("/", "_").replace("=", "_").replace("-", "_")
    return Path(__file__).with_name(f"bot_status_{safe}_{interval}.json")

def _portfolio_status_path(interval: str) -> Path:
    return Path(__file__).with_name(f"bot_status_portfolio_top5_{interval}.json")

STATUS_PATH = _status_path(ticker, interval)

def _bot_loop(symbol: str, period: str, interval: str, status_path: Path, start_capital: float = 10000.0):
    capital = start_capital
    pos = 0
    entry = 0.0
    trades = []
    last_trade_i = -10**9
    while True:
        try:
            d = add_features(load_data(symbol, period, interval), interval)
            sh, rb, rs, vm, _, _, _, _ = auto_tune(d)
            ctl = auto_controls(d, rb, rs, vm)
            _, sig_live, _, _, _ = score(d.copy(), rb, rs, vm, cooldown=ctl["cooldown"], fee=ctl["fee"], slippage=ctl["slippage"], news_blackout=ctl["news"])
            d["signal"] = sig_live
            i = len(d)-1
            s_now = int(d["signal"].iloc[i])
            if cooldown_bars > 0 and (i - last_trade_i) <= cooldown_bars:
                s_now = 0
            last = d.iloc[-1]
            s_now, ai_risk_mult, ai_reason = ai_filter_signal(s_now, last, {"mode":"single_bot"})
            price = float(last["close"])

            if pos != 0 and s_now == -pos:
                vol_now = float(last.get("vol", 0.01)) if pd.notna(last.get("vol", 0.01)) else 0.01
                dyn_risk = max(0.002, min(ctl["risk"] * ai_risk_mult, 0.01 / max(vol_now, 1e-6)))
                pnl = (price - entry) / entry * pos * (capital * dyn_risk * 5)
                capital += pnl
                trades.append(pnl)
                pos = 0
            if pos == 0 and s_now != 0:
                pos = s_now
                entry = price
                last_trade_i = i

            cum_pnl = capital - start_capital
            cum_pct = (capital / start_capital - 1) * 100
            status = {
                "ts": str(d.index[-1]),
                "symbol": symbol,
                "capital": round(capital, 2),
                "cum_pnl": round(cum_pnl, 2),
                "cum_pct": round(cum_pct, 2),
                "controls": ctl,
                "ai_reason": ai_reason
            }
            status_path.write_text(json.dumps(status, indent=2))
        except Exception as e:
            status_path.write_text(json.dumps({"error": str(e), "symbol": symbol, "interval": interval}, indent=2))

        sleep_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
        sleep_sec = sleep_map.get(interval, 300)
        time.sleep(sleep_sec)

def _portfolio_bot_loop(period: str, interval: str, status_path: Path, start_capital: float = 10000.0):
    capital = start_capital
    positions = {}
    entries = {}
    asset_pnl = {}
    while True:
        try:
            rows = []
            for a in PORTF_UNIVERSE:
                try:
                    d = add_features(load_data(a, period, interval), interval)
                    if d is None or d.empty or len(d) < 120:
                        continue
                    sh, rb, rs, vm, *_ = auto_tune(d)
                    ctl = auto_controls(d, rb, rs, vm)
                    _, sig_live, _, _, _ = score(d.copy(), rb, rs, vm, cooldown=ctl["cooldown"], fee=ctl["fee"], slippage=ctl["slippage"], news_blackout=ctl["news"])
                    s_now = int(sig_live.iloc[-1])
                    px = float(d['close'].iloc[-1])
                    s_now, ai_risk_mult, _ = ai_filter_signal(s_now, d.iloc[-1], {"mode":"portfolio", "asset":a})
                    rows.append((a, sh, s_now, px, ai_risk_mult))
                except Exception:
                    continue

            if rows:
                top = sorted(rows, key=lambda x: x[1], reverse=True)[:5]
                active = [r for r in top if r[2] != 0]
                avg_risk = np.mean([max(0.003, min(0.02, 0.01 * r[4])) for r in active]) if active else 0.01
                alloc = (capital * avg_risk * 5 / max(1, len(active)))

                # close positions not in active or flipped
                for a in list(positions.keys()):
                    row = next((r for r in top if r[0] == a), None)
                    if row is None or row[2] == 0 or row[2] == -positions[a]:
                        px = row[3] if row else entries[a]
                        pnl = (px - entries[a]) / entries[a] * positions[a] * alloc
                        capital += pnl
                        asset_pnl[a] = asset_pnl.get(a, 0.0) + pnl
                        positions.pop(a, None)
                        entries.pop(a, None)

                # open/update active
                for a, sh, s_now, px, ai_rm in active:
                    if a not in positions:
                        positions[a] = s_now
                        entries[a] = px

            status = {
                "ts": str(pd.Timestamp.utcnow()),
                "mode": "portfolio_top5",
                "capital": round(capital, 2),
                "cum_pnl": round(capital - start_capital, 2),
                "cum_pct": round((capital / start_capital - 1) * 100, 2),
                "active_positions": len(positions),
                "weights": {a: round(1/max(1,len(positions)),3) for a in positions.keys()},
                "asset_pnl": {k: round(v,2) for k,v in asset_pnl.items()}
            }
            status_path.write_text(json.dumps(status, indent=2))
        except Exception as e:
            status_path.write_text(json.dumps({"error": str(e), "mode": "portfolio_top5", "interval": interval}, indent=2))

        sleep_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
        time.sleep(sleep_map.get(interval, 300))

def start_embedded_bot(symbol: str, period: str, interval: str):
    mode = "portfolio" if portfolio_mode else "single"
    key = f"{mode}|{symbol}|{period}|{interval}"
    if st.session_state.get("bot_key") != key:
        st.session_state["bot_key"] = key
        if portfolio_mode:
            sp = _portfolio_status_path(interval)
            st.session_state["bot_status_path"] = str(sp)
            t = threading.Thread(target=_portfolio_bot_loop, args=(period, interval, sp), daemon=True)
        else:
            sp = _status_path(symbol, interval)
            st.session_state["bot_status_path"] = str(sp)
            t = threading.Thread(target=_bot_loop, args=(symbol, period, interval, sp), daemon=True)
        t.start()
    return True



# adjust period for Yahoo intraday limits
if interval == "1m" and period in ["2y", "5y", "1y"]:
    period = "7d"
elif interval in ["5m", "15m", "30m"] and period in ["2y", "5y"]:
    period = "60d"
@st.cache_data(ttl=900, show_spinner=False)
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
    # higher timeframe confirmation
    htf = "4h" if interval in ["1m","5m","15m","30m","1h"] else "1d"
    d2 = df["close"].resample(htf).last().dropna().to_frame("close")
    d2["ema_hf_fast"] = d2["close"].ewm(span=50, adjust=False).mean()
    d2["ema_hf_slow"] = d2["close"].ewm(span=200, adjust=False).mean()
    htrend = (d2["ema_hf_fast"] > d2["ema_hf_slow"]).reindex(df.index, method="ffill").fillna(False)
    df["htf_up"] = htrend.astype(int)
    df["regime"] = df.apply(regime_label, axis=1)

    return df.dropna()


def score(df, rsi_buy, rsi_sell, vol_min, cooldown=3, fee=5.0, slippage=8.0, news_blackout=True, ai_apply=False):
    # Base trend signal
    trend_sig = pd.Series(0, index=df.index)
    trend_sig[(df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > rsi_buy) & (df["vol"] > vol_min)] = 1
    trend_sig[(df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < rsi_sell) & (df["vol"] > vol_min)] = -1

    # Ensemble components
    mr_sig = pd.Series(0, index=df.index)
    mr_sig[(df["rsi"] < 30) & (df["ema_fast"] > df["ema_slow"])] = 1
    mr_sig[(df["rsi"] > 70) & (df["ema_fast"] < df["ema_slow"])] = -1
    roll_hi = df["close"].rolling(20).max().shift(1)
    roll_lo = df["close"].rolling(20).min().shift(1)
    br_sig = pd.Series(0, index=df.index)
    br_sig[df["close"] > roll_hi] = 1
    br_sig[df["close"] < roll_lo] = -1

    vote = trend_sig + mr_sig + br_sig
    sig = pd.Series(0, index=df.index)
    sig[vote >= 2] = 1
    sig[vote <= -2] = -1
    sig = sig.where(sig != 0, trend_sig)

    # Multi-timeframe confirmation
    sig = sig.where(~((sig == 1) & (df["htf_up"] == 0)), 0)
    sig = sig.where(~((sig == -1) & (df["htf_up"] == 1)), 0)

    # Regime/vol filters
    sig = sig.where(df["vol"] > max(vol_min * 0.8, 0.001), 0)
    sig = sig.where(df["trend_spread"].abs() > 0.0015, 0)

    # Optional news blackout + execution realism controls from sidebar
    if news_blackout:
        sig = sig.where(~blackout_mask(df.index), 0)

    # Cooldown to reduce overtrading
    if cooldown > 0:
        vals = sig.values.copy()
        last_trade = -10**9
        for i in range(len(vals)):
            if vals[i] != 0:
                if i - last_trade <= cooldown:
                    vals[i] = 0
                else:
                    last_trade = i
        sig = pd.Series(vals, index=df.index)

    # AI filter (low-token, optional)
    if ai_apply and AI_ENABLED:
        vals = sig.values.copy()
        conf = np.ones(len(vals), dtype=float)
        for i in range(len(vals)):
            if vals[i] == 0:
                continue
            ai_sig, ai_rm, _ = ai_filter_signal(int(vals[i]), df.iloc[i], {"mode":"score"})
            vals[i] = ai_sig
            conf[i] = ai_rm
        sig = pd.Series(vals, index=df.index)
        ai_conf = pd.Series(conf, index=df.index)
    else:
        ai_conf = pd.Series(1.0, index=df.index)

    # Confidence sizing + costs
    confidence = (vote.abs() / 3.0).clip(0.34, 1.0) * ai_conf
    pos = sig.replace(0, np.nan).ffill().fillna(0)
    gross = pos.shift(1).fillna(0) * df["ret"].fillna(0) * confidence
    turnover = pos.diff().abs().fillna(0)
    cost = turnover * ((fee + slippage) / 10000.0)
    sret = gross - cost

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




def auto_controls(df, rb, rs, vm):
    best = (-1e9, 3, 0.01, 5.0, 8.0, True)
    vol_med = float(df["vol"].median()) if "vol" in df else 0.005
    fee_cands = [max(1.0, round(vol_med*12000,1)), 3.0, 5.0, 8.0]
    slip_cands = [max(2.0, round(vol_med*18000,1)), 5.0, 8.0, 12.0]
    for cd in [1, 2, 3, 5]:
        for risk in [0.005, 0.007, 0.01, 0.015]:
            for fee in sorted(set(fee_cands)):
                for slip in sorted(set(slip_cands)):
                    for nb in [False, True]:
                        sh, *_ = score(df.copy(), rb, rs, vm, cooldown=cd, fee=fee, slippage=slip, news_blackout=nb)
                        # penalize expensive setup a bit
                        obj = sh - (fee + slip) * 0.01
                        if obj > best[0]:
                            best = (obj, cd, risk, fee, slip, nb)
    return {"cooldown": best[1], "risk": best[2], "fee": best[3], "slippage": best[4], "news": best[5]}


def run_reliability(df, rb, rs, vm):
    if len(df) <= 200:
        return {"hit":0.0,"rel_return":0.0,"rel_dd":0.0,"rb":rb,"rs":rs,"vm":vm,"wf_folds":0,"wf_avg":0.0,"test_df":None}
    split_idx = max(1, len(df) - max(30, len(df)//12))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    sh_t, rb_t, rs_t, vm_t, *_ = auto_tune(train_df)
    _, sig_test, sret_test, eq_test, dd_test = score(test_df.copy(), rb_t, rs_t, vm_t, cooldown=cooldown_bars, fee=fee_bps, slippage=slippage_bps, news_blackout=use_news_blackout, ai_apply=True)
    test_df['sig_test']=sig_test; test_df['eq_test']=eq_test; test_df['dd_test']=dd_test
    m=test_df['sig_test']!=0
    hit=float((((test_df['sig_test'].shift(1)*test_df['ret'])>0)&m).sum()/m.sum()*100) if m.any() else 0.0
    rel_return=float((test_df['eq_test'].iloc[-1]-1)*100)
    rel_dd=float(test_df['dd_test'].min()*100)
    wf_folds=0; wf_scores=[]; chunk=max(30,len(df)//10)
    for st_i in range(max(60,chunk), len(df)-chunk, chunk):
        tr=df.iloc[:st_i].copy(); te=df.iloc[st_i:st_i+chunk].copy()
        if len(tr)<80 or len(te)<20: continue
        _, rbw, rsw, vmw, *_ = auto_tune(tr)
        _, _, _, eq_w, _ = score(te.copy(), rbw, rsw, vmw, cooldown=cooldown_bars, fee=fee_bps, slippage=slippage_bps, news_blackout=use_news_blackout, ai_apply=True)
        wf_scores.append((eq_w.iloc[-1]-1)*100); wf_folds+=1
    wf_avg=float(np.mean(wf_scores)) if wf_scores else 0.0
    return {"hit":hit,"rel_return":rel_return,"rel_dd":rel_dd,"rb":rb_t,"rs":rs_t,"vm":vm_t,"wf_folds":wf_folds,"wf_avg":wf_avg,"test_df":test_df}


def render_portfolio_plots(status_path):
    if not portfolio_mode or not status_path.exists():
        return
    try:
        bs = json.loads(status_path.read_text())
        ap = bs.get("asset_pnl", {})
        wt = bs.get("weights", {})
        if ap:
            st.subheader("Portfolio redistribution & P/L by asset")
            cA, cB = st.columns(2)
            with cA:
                figp = go.Figure(go.Bar(x=list(ap.keys()), y=list(ap.values()), name="P/L EUR"))
                figp.update_layout(height=320, title="Profit/Loss by asset")
                st.plotly_chart(figp, use_container_width=True)
            with cB:
                if wt:
                    figw = go.Figure(go.Pie(labels=list(wt.keys()), values=list(wt.values()), hole=0.45))
                    figw.update_layout(height=320, title="Current redistribution weights")
                    st.plotly_chart(figw, use_container_width=True)
    except Exception:
        pass

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
        mode_txt = "Portfolio Top-5" if portfolio_mode else f"{ticker}"
        st.sidebar.caption(f"Mode: {mode_txt} / {interval}")

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

ctl = auto_controls(df, rb, rs, vm)
cooldown_bars = ctl["cooldown"]
max_risk_pct = ctl["risk"]
fee_bps = ctl["fee"]
slippage_bps = ctl["slippage"]
use_news_blackout = ctl["news"]

# Full pipeline score uses all configured options (ensemble, MTF, costs, blackout, cooldown)
_, sig_full, sret, equity, dd = score(df.copy(), rb, rs, vm, cooldown=cooldown_bars, fee=fee_bps, slippage=slippage_bps, news_blackout=use_news_blackout)

df["signal"] = sig_full
df["strategy_ret"] = sret
df["equity"] = equity
df["drawdown"] = dd


if df.empty:
    st.error("Dataset is empty after processing.")
    st.stop()

rel = run_reliability(df.copy(), rb, rs, vm)

latest = df.iloc[-1]
base_sig_now = int(latest["signal"])
sig_now, ui_ai_risk_mult, ui_ai_reason = ai_filter_signal(base_sig_now, latest, {"mode":"ui_decision"})
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

if page == "Reliability":
    st.subheader("Reliability check")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Train window", "From ~12 months ago")
    r2.metric("Test window", "Last month")
    r3.metric("Signal hit rate", f"{rel['hit']:.1f}%")
    r4.metric("Test return / DD", f"{rel['rel_return']:+.2f}% / {rel['rel_dd']:.2f}%")
    st.caption(f"WF folds: {rel['wf_folds']} | avg WF return: {rel['wf_avg']:+.2f}%")
    if rel['test_df'] is not None:
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Scatter(x=rel['test_df'].index, y=rel['test_df']['eq_test'], name='Last-month equity (OOS)'))
        fig_rel.update_layout(height=320, title='Reliability equity (same full pipeline incl AI/costs)')
        st.plotly_chart(fig_rel, use_container_width=True)
    render_portfolio_plots(STATUS_PATH)
    st.stop()

st.markdown(f"## Signal now: :{color}[**{decision}**]")
st.caption(f"Auto controls | RSI>{rb}/{rs}, vol>{vm:.3f}, cooldown={cooldown_bars}, risk={max_risk_pct*100:.1f}%, fee={fee_bps:.1f}bps, slippage={slippage_bps:.1f}bps, blackout={use_news_blackout}")

if STATUS_PATH.exists():
    try:
        bs = json.loads(STATUS_PATH.read_text())
        st.info(f"Bot cumulative P/L: {float(bs.get('cum_pnl',0.0)):+.2f} EUR ({float(bs.get('cum_pct',0.0)):+.2f}%)")
    except Exception:
        pass


if portfolio_mode and STATUS_PATH.exists():
    try:
        bs = json.loads(STATUS_PATH.read_text())
        ap = bs.get("asset_pnl", {})
        wt = bs.get("weights", {})
        if ap:
            st.subheader("Portfolio redistribution & P/L by asset")
            cA, cB = st.columns(2)
            with cA:
                figp = go.Figure(go.Bar(x=list(ap.keys()), y=list(ap.values()), name="P/L EUR"))
                figp.update_layout(height=320, title="Profit/Loss by asset")
                st.plotly_chart(figp, use_container_width=True)
            with cB:
                if wt:
                    figw = go.Figure(go.Pie(labels=list(wt.keys()), values=list(wt.values()), hole=0.45))
                    figw.update_layout(height=320, title="Current redistribution weights")
                    st.plotly_chart(figw, use_container_width=True)
    except Exception:
        pass
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"{latest['close']:.2f}")
c2.metric("Proxy return", f"{(df['equity'].iloc[-1]-1)*100:.2f}%")
c3.metric("Max drawdown", f"{df['drawdown'].min()*100:.2f}%")
c4.metric("Signal activity", f"{(df['signal']!=0).mean()*100:.1f}%")

with st.expander("Why this signal?"):
    st.write(reasons)

# hard risk line for paper safety
if df["drawdown"].min() < -0.06:
    st.warning("Risk guard: drawdown exceeded 6% in this backtest slice.")


# simple paper bot simulation (no real money)
start_cap = st.sidebar.number_input("Paper start capital (EUR)", min_value=100.0, value=10000.0, step=100.0)
risk_per_trade = max_risk_pct

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
        vol_now = float(df["vol"].iloc[i]) if not np.isnan(df["vol"].iloc[i]) else 0.01
        dyn_risk = max(0.002, min(risk_per_trade, 0.01 / max(vol_now, 1e-6)))
        pnl = (px - entry) / entry * pos * (capital * dyn_risk * 5)
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

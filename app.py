import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "trading_journal.db"

st.set_page_config(page_title="Free Order-Block Scanner", page_icon="📊", layout="wide")
st.title("📊 Free Order-Block Scanner")
st.caption("Educational research tool • NSE F&O starter universe • Not investment advice")

DEFAULT_SYMBOLS = [
    ("RELIANCE","Energy"),("HDFCBANK","Banking"),("ICICIBANK","Banking"),("SBIN","Banking"),
    ("AXISBANK","Banking"),("KOTAKBANK","Banking"),("INDUSINDBK","Banking"),("BANKBARODA","Banking"),
    ("BAJFINANCE","Financial Services"),("BAJAJFINSV","Financial Services"),("SHRIRAMFIN","Financial Services"),
    ("LICI","Financial Services"),("SBILIFE","Financial Services"),("HDFCLIFE","Financial Services"),
    ("INFY","IT"),("TCS","IT"),("WIPRO","IT"),("HCLTECH","IT"),("TECHM","IT"),("LTIM","IT"),
    ("LT","Capital Goods"),("ADANIENT","Diversified"),("ADANIPORTS","Infrastructure"),
    ("BHARTIARTL","Telecom"),("ITC","FMCG"),("HINDUNILVR","FMCG"),("TATACONSUM","FMCG"),
    ("ASIANPAINT","Consumer"),("TITAN","Consumer"),("MARUTI","Auto"),("M&M","Auto"),
    ("TATAMOTORS","Auto"),("EICHERMOT","Auto"),("HEROMOTOCO","Auto"),("BAJAJ-AUTO","Auto"),
    ("SUNPHARMA","Pharma"),("DRREDDY","Pharma"),("CIPLA","Pharma"),("DIVISLAB","Pharma"),
    ("APOLLOHOSP","Healthcare"),("TATASTEEL","Metals"),("JSWSTEEL","Metals"),("HINDALCO","Metals"),
    ("COALINDIA","Mining"),("NTPC","Power"),("POWERGRID","Power"),("ONGC","Energy"),
    ("BPCL","Energy"),("GRASIM","Materials"),("ULTRACEMCO","Cement"),("DLF","Real Estate"),
    ("BEL","Defence"),("HAL","Defence"),("TRENT","Retail"),("INDIGO","Aviation"),
    ("DMART","Retail"),("PIDILITIND","Materials"),("GODREJCP","FMCG"),("IOC","Energy")
]

def ensure_symbols():
    p = APP_DIR / "symbols.csv"
    if not p.exists():
        pd.DataFrame(DEFAULT_SYMBOLS, columns=["symbol", "sector"]).to_csv(p, index=False)
    return pd.read_csv(p)

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trade_date TEXT, symbol TEXT,
            side TEXT, setup TEXT, entry REAL, stop REAL, target REAL,
            exit_price REAL, quantity INTEGER, notes TEXT, status TEXT DEFAULT 'Open'
        )""")

init_db()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(symbol, period, interval):
    ticker = symbol.strip().upper() + ".NS"
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                     progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].copy().dropna()
    df.index = pd.to_datetime(df.index)
    return df

def add_indicators(df):
    x = df.copy()
    prev_close = x["Close"].shift(1)
    tr = pd.concat([
        x["High"] - x["Low"],
        (x["High"] - prev_close).abs(),
        (x["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(14).mean()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    x["VolAvg20"] = x["Volume"].rolling(20).mean()
    return x

def confirmed_swings(x, left=3, right=3):
    """A pivot is confirmed only after `right` later candles have closed."""
    highs, lows = [], []
    h = x["High"].to_numpy(dtype=float)
    l = x["Low"].to_numpy(dtype=float)
    for i in range(left, len(x) - right):
        if h[i] >= np.max(h[i-left:i+right+1]) and np.count_nonzero(h[i] == h[i-left:i+right+1]) == 1:
            highs.append((i, float(h[i])))
        if l[i] <= np.min(l[i-left:i+right+1]) and np.count_nonzero(l[i] == l[i-left:i+right+1]) == 1:
            lows.append((i, float(l[i])))
    return highs, lows

def analyse(symbol, sector, raw):
    if raw.empty or len(raw) < 70:
        return None
    x = add_indicators(raw)
    highs, lows = confirmed_swings(x)
    if not highs or not lows:
        return None

    last = x.iloc[-1]
    prev = x.iloc[-2]
    close = float(last.Close)
    atr = float(last.ATR) if pd.notna(last.ATR) and last.ATR > 0 else close * 0.02
    vol_ratio = float(last.Volume / last.VolAvg20) if pd.notna(last.VolAvg20) and last.VolAvg20 > 0 else np.nan
    trend_up = close > float(last.EMA20) > float(last.EMA50)
    trend_dn = close < float(last.EMA20) < float(last.EMA50)

    # Most recent confirmed swing levels that existed before the current candle.
    prior_highs = [(i, p) for i, p in highs if i < len(x)-1]
    prior_lows = [(i, p) for i, p in lows if i < len(x)-1]
    sh_i, sh = prior_highs[-1]
    sl_i, sl = prior_lows[-1]

    bull_break = float(last.Close) > sh and float(prev.Close) <= sh
    bear_break = float(last.Close) < sl and float(prev.Close) >= sl

    # Structure direction immediately before a break, inferred from the last break in the history.
    structure = "Neutral"
    for j in range(20, len(x)-1):
        past_highs = [(i, p) for i, p in highs if i < j]
        past_lows = [(i, p) for i, p in lows if i < j]
        if not past_highs or not past_lows:
            continue
        c = float(x["Close"].iloc[j])
        pc = float(x["Close"].iloc[j-1])
        if c > past_highs[-1][1] and pc <= past_highs[-1][1]:
            structure = "Bullish"
        elif c < past_lows[-1][1] and pc >= past_lows[-1][1]:
            structure = "Bearish"

    bos = "—"
    choch = "—"
    if bull_break:
        if structure == "Bearish":
            choch, bos = "Bullish CHoCH", "—"
        else:
            bos = "Bullish BOS"
        structure = "Bullish"
    elif bear_break:
        if structure == "Bullish":
            choch, bos = "Bearish CHoCH", "—"
        else:
            bos = "Bearish BOS"
        structure = "Bearish"

    # Liquidity sweep: wick takes a confirmed swing and candle closes back inside.
    sweep = "—"
    if float(last.High) > sh and close < sh:
        sweep = "Buy-side"
    elif float(last.Low) < sl and close > sl:
        sweep = "Sell-side"

    # Find the last opposing candle between the broken swing and break candle.
    bull_ob = bear_ob = None
    if bull_break:
        start = max(sh_i + 1, len(x) - 16)
        for i in range(len(x)-2, start-1, -1):
            row = x.iloc[i]
            if row.Close < row.Open:
                bull_ob = (float(row.Low), float(row.High), x.index[i])
                break
    if bear_break:
        start = max(sl_i + 1, len(x) - 16)
        for i in range(len(x)-2, start-1, -1):
            row = x.iloc[i]
            if row.Close > row.Open:
                bear_ob = (float(row.Low), float(row.High), x.index[i])
                break

    # Three-candle Fair Value Gap (FVG); latest unfilled gap is reported.
    bull_fvg = bear_fvg = None
    for i in range(len(x)-1, 1, -1):
        a, c = x.iloc[i-2], x.iloc[i]
        if float(c.Low) > float(a.High):
            bull_fvg = (float(a.High), float(c.Low), x.index[i])
            break
        if float(c.High) < float(a.Low):
            bear_fvg = (float(c.High), float(a.Low), x.index[i])
            break

    score = 0
    score += 2 if trend_up else (-2 if trend_dn else 0)
    score += 2 if bos == "Bullish BOS" or choch == "Bullish CHoCH" else (-2 if bos == "Bearish BOS" or choch == "Bearish CHoCH" else 0)
    score += 1 if close > float(last.EMA20) else (-1 if close < float(last.EMA20) else 0)
    score += 1 if np.isfinite(vol_ratio) and vol_ratio >= 1.5 else 0
    score += 1 if sweep == "Sell-side" else (-1 if sweep == "Buy-side" else 0)
    bias = "Bullish" if score >= 3 else ("Bearish" if score <= -3 else "Neutral")

    side = "LONG" if bias != "Bearish" else "SHORT"
    entry = close
    stop = close - 1.5*atr if side == "LONG" else close + 1.5*atr
    target1 = close + 2*atr if side == "LONG" else close - 2*atr
    target2 = close + 3*atr if side == "LONG" else close - 3*atr

    return {
        "Symbol": symbol, "Sector": sector, "Close": round(close, 2),
        "Change %": round((close/float(prev.Close)-1)*100, 2),
        "Score": score, "Bias": bias, "Volume x": round(vol_ratio, 2) if np.isfinite(vol_ratio) else np.nan,
        "BOS": bos, "CHoCH": choch, "Liquidity sweep": sweep,
        "Swing high": round(sh, 2), "Swing low": round(sl, 2),
        "Entry (reference)": round(entry, 2), "Stop (illustrative)": round(stop, 2),
        "Target 1": round(target1, 2), "Target 2": round(target2, 2),
        "ATR": round(atr, 2), "Bull OB": bull_ob, "Bear OB": bear_ob,
        "Bull FVG": bull_fvg, "Bear FVG": bear_fvg, "data": x
    }

def make_chart(r):
    x = r["data"].tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=x.index, open=x.Open, high=x.High, low=x.Low, close=x.Close, name="Price"))
    fig.add_trace(go.Scatter(x=x.index, y=x.EMA20, name="EMA 20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x.index, y=x.EMA50, name="EMA 50", line=dict(width=1)))
    for key, color, label in [
        ("Bull OB", "rgba(0,180,80,0.22)", "Bullish OB"),
        ("Bear OB", "rgba(220,60,60,0.22)", "Bearish OB"),
        ("Bull FVG", "rgba(0,130,220,0.15)", "Bullish FVG"),
        ("Bear FVG", "rgba(220,150,0,0.15)", "Bearish FVG")
    ]:
        zone = r[key]
        if zone:
            lo, hi, stamp = zone
            fig.add_hrect(y0=lo, y1=hi, fillcolor=color, line_width=0,
                          annotation_text=label, annotation_position="top left")
    fig.add_hline(y=r["Swing high"], line_dash="dot", line_width=1, annotation_text="Last confirmed swing high")
    fig.add_hline(y=r["Swing low"], line_dash="dot", line_width=1, annotation_text="Last confirmed swing low")
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
    return fig

symbols_df = ensure_symbols()
st.sidebar.header("Scanner settings")
interval_label = st.sidebar.selectbox("Candle timeframe", ["1 day", "15 minutes", "5 minutes"], index=0)
interval = {"1 day":"1d", "15 minutes":"15m", "5 minutes":"5m"}[interval_label]
period = {"1d":"1y", "15m":"60d", "5m":"60d"}[interval]
max_symbols = st.sidebar.slider("Number of stocks to scan", 5, min(60, len(symbols_df)), min(25, len(symbols_df)))
st.sidebar.caption("Starter list is editable in symbols.csv. Refresh the F&O list from NSE's official contract information page.")
if st.sidebar.button("Clear cached market data"):
    fetch_history.clear()
scan_button = st.sidebar.button("🔄 Run / refresh scan", type="primary")

if "results" not in st.session_state:
    st.session_state.results = []
if scan_button:
    out = []
    progress = st.progress(0)
    status = st.empty()
    subset = symbols_df.head(max_symbols)
    for i, row in enumerate(subset.itertuples(index=False), start=1):
        status.write(f"Fetching {row.symbol} ({i}/{len(subset)})…")
        try:
            hist = fetch_history(row.symbol, period, interval)
            result = analyse(row.symbol, row.sector, hist)
            if result:
                out.append(result)
        except Exception:
            pass
        progress.progress(i/len(subset))
    st.session_state.results = out
    status.write(f"Scan complete: {len(out)} stocks returned data.")
results = st.session_state.results

tabs = st.tabs(["Scanner", "Order blocks & chart", "Sector & breadth", "OI / volume", "Trading journal", "How it works"])
with tabs[0]:
    st.subheader("Bullish and bearish stock scanner")
    if results:
        table = pd.DataFrame([{k:v for k,v in r.items() if k not in ["data","Bull OB","Bear OB","Bull FVG","Bear FVG"]} for r in results])
        c1,c2,c3 = st.columns(3)
        c1.metric("Bullish", int((table.Bias=="Bullish").sum()))
        c2.metric("Bearish", int((table.Bias=="Bearish").sum()))
        c3.metric("Neutral", int((table.Bias=="Neutral").sum()))
        filt = st.selectbox("Show", ["All","Bullish","Bearish","Neutral"])
        view = table if filt=="All" else table[table.Bias==filt]
        st.dataframe(view.sort_values(["Score","Volume x"], ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download scanner CSV", table.to_csv(index=False).encode(), "scanner_results.csv", "text/csv")
    else:
        st.info("Choose settings on the left and click Run / refresh scan. First run downloads public Yahoo Finance data.")

with tabs[1]:
    st.subheader("Market structure, order blocks & chart")
    if results:
        names = [r["Symbol"] for r in results]
        chosen = st.selectbox("Choose stock", names)
        r = next(z for z in results if z["Symbol"] == chosen)
        st.plotly_chart(make_chart(r), use_container_width=True)
        a,b,c,d = st.columns(4)
        a.metric("Bias", r["Bias"])
        b.metric("BOS", r["BOS"])
        c.metric("CHoCH", r["CHoCH"])
        d.metric("Liquidity sweep", r["Liquidity sweep"])
        st.write("Last confirmed swing high:", r["Swing high"], "| Swing low:", r["Swing low"])
        st.write("Bullish OB:", r["Bull OB"][:2] if r["Bull OB"] else "Not detected")
        st.write("Bearish OB:", r["Bear OB"][:2] if r["Bear OB"] else "Not detected")
        st.write("Bullish FVG:", r["Bull FVG"][:2] if r["Bull FVG"] else "Not detected")
        st.write("Bearish FVG:", r["Bear FVG"][:2] if r["Bear FVG"] else "Not detected")
        st.subheader("Illustrative risk levels — not trade recommendations")
        st.write(f"Reference close: {r['Entry (reference)']} | Stop: {r['Stop (illustrative)']} | T1: {r['Target 1']} | T2: {r['Target 2']}")
        st.caption("Structure uses confirmed 3-left/3-right pivots. A pivot is only known after the right-side candles close. OB and FVG rules are transparent approximations, not a universal institutional definition.")
    else:
        st.info("Run a scan first.")

with tabs[2]:
    st.subheader("Sector strength & market breadth")
    if results:
        t = pd.DataFrame([{k:v for k,v in r.items() if k not in ["data","Bull OB","Bear OB","Bull FVG","Bear FVG"]} for r in results])
        sector = t.groupby("Sector").agg(Stocks=("Symbol","count"), AvgChange=("Change %","mean"),
                    Bullish=("Bias",lambda s:(s=="Bullish").sum()),
                    Bearish=("Bias",lambda s:(s=="Bearish").sum())).reset_index()
        sector["AvgChange"] = sector["AvgChange"].round(2)
        st.dataframe(sector.sort_values("AvgChange",ascending=False), use_container_width=True, hide_index=True)
        st.metric("Market breadth (bullish minus bearish)", int((t.Bias=="Bullish").sum()-(t.Bias=="Bearish").sum()))
        st.caption("Breadth covers only successfully scanned symbols in your current sample.")
    else:
        st.info("Run a scan first.")

with tabs[3]:
    st.subheader("Open interest & volume")
    st.write("Volume expansion is calculated from price history. Yahoo Finance does not provide dependable NSE stock-futures OI history through this free feed.")
    if results:
        t = pd.DataFrame([{k:v for k,v in r.items() if k not in ["data","Bull OB","Bear OB","Bull FVG","Bear FVG"]} for r in results])
        st.dataframe(t[["Symbol","Volume x","Change %","Bias"]].sort_values("Volume x",ascending=False), use_container_width=True, hide_index=True)
    st.markdown("**Optional OI CSV upload**")
    st.caption("Use columns Symbol, OI, Prev_OI. Values must refer to the same futures contract/expiry on comparable consecutive snapshots or sessions.")
    oi_file = st.file_uploader("Upload OI CSV", type=["csv"], key="oi")
    if oi_file:
        oi = pd.read_csv(oi_file)
        required = {"Symbol","OI","Prev_OI"}
        if required.issubset(oi.columns):
            oi["OI Change %"] = (oi.OI/oi.Prev_OI.replace(0,np.nan)-1)*100
            if results:
                prices = pd.DataFrame([{"Symbol":r["Symbol"],"Change %":r["Change %"]} for r in results])
                oi = oi.merge(prices,on="Symbol",how="left")
            oi["Interpretation"] = np.select([
                (oi["Change %"]>0)&(oi["OI Change %"]>0),
                (oi["Change %"]<0)&(oi["OI Change %"]>0),
                (oi["Change %"]>0)&(oi["OI Change %"]<0),
                (oi["Change %"]<0)&(oi["OI Change %"]<0)],
                ["Long build-up","Short build-up","Short covering","Long unwinding"], default="Mixed / unavailable")
            st.dataframe(oi,use_container_width=True,hide_index=True)
        else:
            st.error("CSV must contain Symbol, OI, Prev_OI columns.")

with tabs[4]:
    st.subheader("Trading journal")
    with st.form("trade_form"):
        c1,c2,c3 = st.columns(3)
        trade_date = c1.date_input("Trade date", date.today())
        symbol = c2.text_input("Symbol", "RELIANCE").upper()
        side = c3.selectbox("Side", ["LONG","SHORT"])
        c4,c5,c6 = st.columns(3)
        setup = c4.selectbox("Setup", ["Bullish OB","Bearish OB","Liquidity sweep","BOS retest","CHoCH","Other"])
        entry = c5.number_input("Entry", min_value=0.0, value=0.0, step=0.05)
        stop = c6.number_input("Stop loss", min_value=0.0, value=0.0, step=0.05)
        c7,c8,c9 = st.columns(3)
        target = c7.number_input("Target", min_value=0.0, value=0.0, step=0.05)
        qty = c8.number_input("Quantity", min_value=1, value=1, step=1)
        exit_price = c9.number_input("Exit price (0 if open)", min_value=0.0, value=0.0, step=0.05)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Save trade")
        if submitted:
            with sqlite3.connect(DB_PATH) as con:
                con.execute("INSERT INTO trades(trade_date,symbol,side,setup,entry,stop,target,exit_price,quantity,notes,status) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (str(trade_date),symbol,side,setup,entry,stop,target,exit_price,int(qty),notes,"Closed" if exit_price>0 else "Open"))
            st.success("Trade saved.")
    with sqlite3.connect(DB_PATH) as con:
        journal = pd.read_sql_query("SELECT * FROM trades ORDER BY trade_date DESC,id DESC", con)
    if not journal.empty:
        journal["P&L"] = np.where(journal.exit_price>0,
            np.where(journal.side=="LONG",(journal.exit_price-journal.entry)*journal.quantity,
                     (journal.entry-journal.exit_price)*journal.quantity), np.nan)
        st.dataframe(journal,use_container_width=True,hide_index=True)
        st.download_button("Download journal CSV",journal.to_csv(index=False).encode(),"trading_journal.csv","text/csv")
    else:
        st.info("No trades recorded yet.")

with tabs[5]:
    st.subheader("Methodology, limits and safety")
    st.markdown("""
- **Universe:** starter sample of NSE F&O underlyings in `symbols.csv`; this is not guaranteed to be the complete current F&O list.
- **Price feed:** `yfinance` accesses Yahoo Finance public endpoints. It is unofficial, may be delayed, rate-limited, incomplete or unavailable, and is not an exchange-authorised live feed.
- **Trend:** EMA20 / EMA50 alignment.
- **Swing structure:** confirmed 3-left / 3-right pivot highs and lows. A pivot is not confirmed until three later candles have closed.
- **BOS / CHoCH:** candle close through the latest confirmed swing level; classification depends on the prior detected structure direction.
- **Liquidity sweep:** wick moves beyond the latest confirmed swing and closes back inside.
- **Order block:** last opposite-colour candle in the recent segment between the swing and a detected structure break. This is a rule-based approximation.
- **FVG:** three-candle price gap approximation.
- **Risk levels:** reference close, 1.5 ATR stop, 2 ATR target 1 and 3 ATR target 2. These are illustrative, not recommendations or guaranteed outcomes.
- **OI:** import a properly matched futures OI CSV. Do not mix expiries or compare unlike contracts.
- **Sector breadth:** only successfully scanned symbols in your current sample, not the entire NSE market.
- **No auto-trading:** this app never places orders or connects to your broker.
""")
    st.markdown("NSE official F&O reports and contract information: https://www.nseindia.com/all-reports-derivatives")
    st.warning("Do not trade solely on scanner output. Validate data, contract expiry, liquidity, slippage, corporate actions and position sizing. Paper-trade and backtest rules first.")

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
    prev = x["Close"].shift(1)
    tr = pd.concat([x["High"]-x["Low"], (x["High"]-prev).abs(),
                    (x["Low"]-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(14).mean()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    x["VolAvg20"] = x["Volume"].rolling(20).mean()
    return x

def confirmed_swings(x, left=3, right=3):
    highs, lows = [], []
    h, l = x["High"].to_numpy(float), x["Low"].to_numpy(float)
    for i in range(left, len(x)-right):
        hw, lw = h[i-left:i+right+1], l[i-left:i+right+1]
        if h[i] == np.max(hw) and np.count_nonzero(hw == h[i]) == 1:
            highs.append((i, float(h[i])))
        if l[i] == np.min(lw) and np.count_nonzero(lw == l[i]) == 1:
            lows.append((i, float(l[i])))
    return highs, lows

def analyse(symbol, sector, raw):
    if raw.empty or len(raw) < 70:
        return None
    x = add_indicators(raw)
    highs, lows = confirmed_swings(x)
    if not highs or not lows:
        return None
    last, prev = x.iloc[-1], x.iloc[-2]
    close = float(last.Close)
    atr = float(last.ATR) if pd.notna(last.ATR) and last.ATR > 0 else close*.02
    vr = float(last.Volume/last.VolAvg20) if pd.notna(last.VolAvg20) and last.VolAvg20 > 0 else np.nan
    sh_i, sh = [(i,p) for i,p in highs if i < len(x)-1][-1]
    sl_i, sl = [(i,p) for i,p in lows if i < len(x)-1][-1]
    bull_break = close > sh and float(prev.Close) <= sh
    bear_break = close < sl and float(prev.Close) >= sl
    structure = "Neutral"
    for j in range(20, len(x)-1):
        ph = [(i,p) for i,p in highs if i < j]
        pl = [(i,p) for i,p in lows if i < j]
        if not ph or not pl:
            continue
        c, pc = float(x.Close.iloc[j]), float(x.Close.iloc[j-1])
        if c > ph[-1][1] and pc <= ph[-1][1]:
            structure = "Bullish"
        elif c < pl[-1][1] and pc >= pl[-1][1]:
            structure = "Bearish"
    bos, choch = "—", "—"
    if bull_break:
        bos, choch, structure = ("—", "Bullish CHoCH", "Bullish") if structure == "Bearish" else ("Bullish BOS", "—", "Bullish")
    elif bear_break:
        bos, choch, structure = ("—", "Bearish CHoCH", "Bearish") if structure == "Bullish" else ("Bearish BOS", "—", "Bearish")
    sweep = "—"
    if float(last.High) > sh and close < sh:
        sweep = "Buy-side"
    elif float(last.Low) < sl and close > sl:
        sweep = "Sell-side"
    bull_ob = bear_ob = None
    if bull_break:
        for i in range(len(x)-2, max(sh_i, len(x)-17), -1):
            row = x.iloc[i]
            if row.Close < row.Open:
                bull_ob = (float(row.Low), float(row.High), x.index[i]); break
    if bear_break:
        for i in range(len(x)-2, max(sl_i, len(x)-17), -1):
            row = x.iloc[i]
            if row.Close > row.Open:
                bear_ob = (float(row.Low), float(row.High), x.index[i]); break
    bull_fvg = bear_fvg = None
    for i in range(len(x)-1, 1, -1):
        a, c = x.iloc[i-2], x.iloc[i]
        if float(c.Low) > float(a.High):
            bull_fvg = (float(a.High), float(c.Low), x.index[i]); break
        if float(c.High) < float(a.Low):
            bear_fvg = (float(c.High), float(a.Low), x.index[i]); break
    trend_up = close > float(last.EMA20) > float(last.EMA50)
    trend_dn = close < float(last.EMA20) < float(last.EMA50)
    score = (2 if trend_up else -2 if trend_dn else 0)
    score += (2 if bos == "Bullish BOS" or choch == "Bullish CHoCH" else -2 if bos == "Bearish BOS" or choch == "Bearish CHoCH" else 0)
    score += (1 if close > float(last.EMA20) else -1 if close < float(last.EMA20) else 0)
    score += (1 if np.isfinite(vr) and vr >= 1.5 else 0)
    score += (1 if sweep == "Sell-side" else -1 if sweep == "Buy-side" else 0)
    bias = "Bullish" if score >= 3 else "Bearish" if score <= -3 else "Neutral"
    side = "LONG" if bias != "Bearish" else "SHORT"
    return {"Symbol":symbol,"Sector":sector,"Close":round(close,2),
        "Change %":round((close/float(prev.Close)-1)*100,2),"Score":score,"Bias":bias,
        "Volume x":round(vr,2) if np.isfinite(vr) else np.nan,"BOS":bos,"CHoCH":choch,
        "Liquidity sweep":sweep,"Swing high":round(sh,2),"Swing low":round(sl,2),
        "Entry (reference)":round(close,2),"Stop (illustrative)":round(close-(1.5*atr) if side=="LONG" else close+(1.5*atr),2),
        "Target 1":round(close+(2*atr) if side=="LONG" else close-(2*atr),2),
        "Target 2":round(close+(3*atr) if side=="LONG" else close-(3*atr),2),
        "ATR":round(atr,2),"Bull OB":bull_ob,"Bear OB":bear_ob,
        "Bull FVG":bull_fvg,"Bear FVG":bear_fvg,"data":x}

def make_chart(r):
    x = r["data"].tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=x.index,open=x.Open,high=x.High,low=x.Low,close=x.Close,name="Price"))
    fig.add_trace(go.Scatter(x=x.index,y=x.EMA20,name="EMA 20",line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x.index,y=x.EMA50,name="EMA 50",line=dict(width=1)))
    for key, color, label in [("Bull OB","rgba(0,180,80,0.22)","Bullish OB"),
        ("Bear OB","rgba(220,60,60,0.22)","Bearish OB"),("Bull FVG","rgba(0,130,220,0.15)","Bullish FVG"),
        ("Bear FVG","rgba(220,150,0,0.15)","Bearish FVG")]:
        if r[key]:
            lo,hi,_ = r[key]
            fig.add_hrect(y0=lo,y1=hi,fillcolor=color,line_width=0,annotation_text=label)
    fig.add_hline(y=r["Swing high"],line_dash="dot",line_width=1,annotation_text="Swing high")
    fig.add_hline(y=r["Swing low"],line_dash="dot",line_width=1,annotation_text="Swing low")
    fig.update_layout(height=540,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=30,b=10))
    return fig

def tf_summary(raw):
    if raw is None or raw.empty or len(raw) < 60:
        return {"Trend":"Insufficient data","Structure":"—","Close":None,"EMA20":None,"EMA50":None,"ATR":None,"Data":raw}
    x = add_indicators(raw)
    highs,lows = confirmed_swings(x)
    last,prev = x.iloc[-1],x.iloc[-2]
    close = float(last.Close)
    trend = "Bullish" if close > last.EMA20 > last.EMA50 else "Bearish" if close < last.EMA20 < last.EMA50 else "Mixed"
    structure = "No recent break"
    if highs and lows:
        sh = [p for i,p in highs if i < len(x)-1]
        sl = [p for i,p in lows if i < len(x)-1]
        if sh and close > sh[-1] and float(prev.Close) <= sh[-1]:
            structure = "Bullish BOS"
        elif sl and close < sl[-1] and float(prev.Close) >= sl[-1]:
            structure = "Bearish BOS"
    atr = float(last.ATR) if pd.notna(last.ATR) and last.ATR > 0 else None
    return {"Trend":trend,"Structure":structure,"Close":close,"EMA20":float(last.EMA20),
            "EMA50":float(last.EMA50),"ATR":atr,"Data":x}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_mtf(symbol):
    out = {}
    for name, period, interval in [("Daily","2y","1d"),("Hourly","730d","1h"),("15M","60d","15m")]:
        try:
            out[name] = fetch_history(symbol,period,interval)
        except Exception:
            out[name] = pd.DataFrame()
    hourly = out["Hourly"]
    if not hourly.empty:
        try:
            four = hourly.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            out["4H"] = four
        except Exception:
            out["4H"] = pd.DataFrame()
    else:
        out["4H"] = pd.DataFrame()
    return out

def zone_retest(x, direction):
    if x is None or x.empty or len(x) < 10:
        return None
    # Look for a recent three-candle fair-value gap, then check whether a later candle revisited it.
    start = max(2, len(x)-35)
    for i in range(len(x)-2, start-1, -1):
        a,c = x.iloc[i-2],x.iloc[i]
        if direction == "Bullish" and float(c.Low) > float(a.High):
            lo,hi = float(a.High),float(c.Low)
        elif direction == "Bearish" and float(c.High) < float(a.Low):
            lo,hi = float(c.High),float(a.Low)
        else:
            continue
        later = x.iloc[i+1:]
        if later.empty:
            continue
        touched = ((later.Low <= hi) & (later.High >= lo)).any()
        last = x.iloc[-1]
        held = float(last.Close) > lo if direction == "Bullish" else float(last.Close) < hi
        if touched and held:
            return (lo,hi)
    return None

def signal_engine(symbol):
    feeds = fetch_mtf(symbol)
    reads = {k:tf_summary(v) for k,v in feeds.items()}
    daily, four, hourly, fifteen = (reads.get(k,{}) for k in ["Daily","4H","Hourly","15M"])
    if any(r.get("Trend") == "Insufficient data" for r in [daily,four,hourly,fifteen]):
        return reads, {"Status":"Insufficient data","Direction":"—","Score":0,"Reasons":["One or more timeframes did not return enough candles."],"Entry":None,"Stop":None,"T1":None,"T2":None}
    direction = "Bullish" if daily["Trend"] == four["Trend"] == "Bullish" else "Bearish" if daily["Trend"] == four["Trend"] == "Bearish" else "Mixed"
    reasons = [f"Daily trend: {daily['Trend']}",f"4H trend: {four['Trend']}",
               f"1H structure: {hourly['Structure']}",f"15M trend: {fifteen['Trend']}"]
    score = 0
    if direction == "Bullish": score += 2
    elif direction == "Bearish": score -= 2
    if direction == "Bullish" and hourly["Structure"] == "Bullish BOS": score += 2
    elif direction == "Bearish" and hourly["Structure"] == "Bearish BOS": score -= 2
    if direction == "Bullish" and fifteen["Trend"] == "Bullish": score += 1
    elif direction == "Bearish" and fifteen["Trend"] == "Bearish": score -= 1
    x = fifteen["Data"]
    bull_retest = zone_retest(x,"Bullish")
    bear_retest = zone_retest(x,"Bearish")
    last = x.iloc[-1]
    if direction == "Bullish" and bull_retest:
        score += 2; reasons.append(f"15M bullish FVG retest held: {bull_retest[0]:.2f}–{bull_retest[1]:.2f}")
    elif direction == "Bearish" and bear_retest:
        score -= 2; reasons.append(f"15M bearish FVG retest held: {bear_retest[0]:.2f}–{bear_retest[1]:.2f}")
    else:
        reasons.append("No confirmed matching 15M FVG retest.")
    highs,lows = confirmed_swings(x)
    if highs and lows:
        sh = [p for i,p in highs if i < len(x)-1]
        sl = [p for i,p in lows if i < len(x)-1]
        if sl and float(last.Low) < sl[-1] and float(last.Close) > sl[-1] and direction == "Bullish":
            score += 1; reasons.append("15M sell-side liquidity sweep detected.")
        elif sh and float(last.High) > sh[-1] and float(last.Close) < sh[-1] and direction == "Bearish":
            score -= 1; reasons.append("15M buy-side liquidity sweep detected.")
    side = "LONG" if direction == "Bullish" else "SHORT" if direction == "Bearish" else "NONE"
    aligned = (side=="LONG" and score>=5) or (side=="SHORT" and score<=-5)
    retest_ok = bool(bull_retest if side=="LONG" else bear_retest) if side!="NONE" else False
    if aligned and retest_ok:
        status = "Potential long setup" if side=="LONG" else "Potential short setup"
    elif direction != "Mixed" and abs(score) >= 2:
        status = "Watch — confirmation incomplete"
    else:
        status = "No setup"
    atr = fifteen.get("ATR")
    entry = stop = t1 = t2 = None
    if status.startswith("Potential") and atr:
        entry = float(last.Close)
        stop = entry - 1.5*atr if side=="LONG" else entry + 1.5*atr
        t1 = entry + 2*atr if side=="LONG" else entry - 2*atr
        t2 = entry + 3*atr if side=="LONG" else entry - 3*atr
    return reads, {"Status":status,"Direction":side,"Score":score,"Reasons":reasons,
                   "Entry":entry,"Stop":stop,"T1":t1,"T2":t2}

symbols_df = ensure_symbols()
st.sidebar.header("Scanner settings")
interval_label = st.sidebar.selectbox("Candle timeframe",["1 day","15 minutes","5 minutes"],index=0)
interval = {"1 day":"1d","15 minutes":"15m","5 minutes":"5m"}[interval_label]
period = {"1d":"1y","15m":"60d","5m":"60d"}[interval]
max_symbols = st.sidebar.slider("Number of stocks to scan",5,min(60,len(symbols_df)),min(25,len(symbols_df)))
st.sidebar.caption("Starter list is editable in symbols.csv. Check NSE official contract information for the current F&O universe.")
if st.sidebar.button("Clear cached market data"):
    fetch_history.clear()
    fetch_mtf.clear()
scan_button = st.sidebar.button("🔄 Run / refresh scan",type="primary")
if "results" not in st.session_state:
    st.session_state.results = []
if scan_button:
    out=[]; progress=st.progress(0); status=st.empty()
    subset=symbols_df.head(max_symbols)
    for i,row in enumerate(subset.itertuples(index=False),start=1):
        status.write(f"Fetching {row.symbol} ({i}/{len(subset)})…")
        try:
            res=analyse(row.symbol,row.sector,fetch_history(row.symbol,period,interval))
            if res: out.append(res)
        except Exception:
            pass
        progress.progress(i/len(subset))
    st.session_state.results=out
    status.write(f"Scan complete: {len(out)} stocks returned data.")
results=st.session_state.results
tabs=st.tabs(["Scanner","Order blocks & chart","Multi-timeframe","Signal engine","Sector & breadth","OI / volume","Trading journal","How it works"])

with tabs[0]:
    st.subheader("Bullish and bearish stock scanner")
    if results:
        table=pd.DataFrame([{k:v for k,v in r.items() if k not in ["data","Bull OB","Bear OB","Bull FVG","Bear FVG"]} for r in results])
        a,b,c=st.columns(3)
        a.metric("Bullish",int((table.Bias=="Bullish").sum()))
        b.metric("Bearish",int((table.Bias=="Bearish").sum()))
        c.metric("Neutral",int((table.Bias=="Neutral").sum()))
        filt=st.selectbox("Show",["All","Bullish","Bearish","Neutral"])
        view=table if filt=="All" else table[table.Bias==filt]
        st.dataframe(view.sort_values(["Score","Volume x"],ascending=False),use_container_width=True,hide_index=True)
        st.download_button("Download scanner CSV",table.to_csv(index=False).encode(),"scanner_results.csv","text/csv")
    else: st.info("Choose settings on the left and click Run / refresh scan.")

with tabs[1]:
    st.subheader("Market structure, order blocks & chart")
    if results:
        chosen=st.selectbox("Choose stock",[r["Symbol"] for r in results],key="chart_stock")
        r=next(z for z in results if z["Symbol"]==chosen)
        st.plotly_chart(make_chart(r),use_container_width=True)
        a,b,c,d=st.columns(4)
        a.metric("Bias",r["Bias"]); b.metric("BOS",r["BOS"]); c.metric("CHoCH",r["CHoCH"]); d.metric("Liquidity sweep",r["Liquidity sweep"])
        st.write("Last confirmed swing high:",r["Swing high"],"| Swing low:",r["Swing low"])
        for label,key in [("Bullish OB","Bull OB"),("Bearish OB","Bear OB"),("Bullish FVG","Bull FVG"),("Bearish FVG","Bear FVG")]:
            st.write(label+":",r[key][:2] if r[key] else "Not detected")
        st.caption("OB and FVG are rule-based approximations, not universal definitions.")
    else: st.info("Run a scan first.")

with tabs[2]:
    st.subheader("Multi-timeframe trend check")
    mtf_symbol=st.selectbox("Choose NSE stock",symbols_df.symbol.tolist(),key="mtf_stock")
    if st.button("Analyse all timeframes",key="mtf_button"):
        with st.spinner("Downloading available candles…"):
            feeds=fetch_mtf(mtf_symbol)
            st.session_state["mtf_view"]={k:tf_summary(v) for k,v in feeds.items()}
    mtf_view=st.session_state.get("mtf_view")
    if mtf_view:
        rows=[{"Timeframe":k,"Trend":v["Trend"],"Structure":v["Structure"],"Latest close":round(v["Close"],2) if v["Close"] is not None else "—"} for k,v in mtf_view.items()]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.caption("4H candles are resampled from Yahoo hourly candles; they are approximate, not exchange-native 4H candles.")

with tabs[3]:
    st.subheader("Advanced signal engine")
    st.warning("Research-only. A potential setup is not a buy/sell recommendation. Data may be delayed, incomplete or unavailable.")
    sig_symbol=st.selectbox("Select stock for signal check",symbols_df.symbol.tolist(),key="signal_stock")
    if st.button("Run signal analysis",type="primary",key="signal_button"):
        with st.spinner("Checking Daily, 4H, 1H and 15M candles…"):
            try:
                reads,signal=signal_engine(sig_symbol)
                st.session_state["signal_result"]=(sig_symbol,reads,signal)
            except Exception as e:
                st.session_state["signal_result"]=(sig_symbol,{},{"Status":"Analysis error","Direction":"—","Score":0,"Reasons":[str(e)],"Entry":None,"Stop":None,"T1":None,"T2":None})
    saved=st.session_state.get("signal_result")
    if saved:
        stock,reads,signal=saved
        st.markdown(f"**Stock:** {stock}")
        st.subheader(signal["Status"])
        st.write("Direction:",signal["Direction"],"| Confluence score:",signal["Score"])
        st.dataframe(pd.DataFrame([{"Timeframe":k,"Trend":v["Trend"],"Structure":v["Structure"]} for k,v in reads.items()]),use_container_width=True,hide_index=True)
        st.markdown("**Why this result?**")
        for reason in signal["Reasons"]: st.write("• "+reason)
        if signal["Entry"] is not None:
            st.write(f"Illustrative entry: ₹{signal['Entry']:.2f} | Stop: ₹{signal['Stop']:.2f} | Target 1: ₹{signal['T1']:.2f} | Target 2: ₹{signal['T2']:.2f}")
        else:
            st.info("No entry, stop or target is displayed unless the multi-timeframe conditions meet the potential-setup rules.")
    else: st.info("Select a stock and tap Run signal analysis.")

with tabs[4]:
    st.subheader("Sector strength & market breadth")
    if results:
        t=pd.DataFrame([{k:v for k,v in r.items() if k not in ["data","Bull OB","Bear OB","Bull FVG","Bear FVG"]} for r in results])
        sector=t.groupby("Sector").agg(Stocks=("Symbol","count"),AvgChange=("Change %","mean"),Bullish=("Bias",lambda s:(s=="Bullish").sum()),Bearish=("Bias",lambda s:(s=="Bearish").sum())).reset_index()
        sector["AvgChange"]=sector["AvgChange"].round(2)
        st.dataframe(sector.sort_values("AvgChange",ascending=False),use_container_width=True,hide_index=True)
        st.metric("Market breadth (bullish minus bearish)",int((t.Bias=="Bullish").sum()-(t.Bias=="Bearish").sum()))
        st.caption("Breadth covers only successfully scanned symbols in your current sample.")
    else: st.info("Run a scan first.")

with tabs[5]:
    st.subheader("Open interest & volume")
    st.write("Volume expansion is calculated from price history. Yahoo Finance does not provide dependable NSE stock-futures OI history through this free feed.")
    if results:
        t=pd.DataFrame([{k:v for k,v in r.items() if k not in ["data","Bull OB","Bear OB","Bull FVG","Bear FVG"]} for r in results])
        st.dataframe(t[["Symbol","Volume x","Change %","Bias"]].sort_values("Volume x",ascending=False),use_container_width=True,hide_index=True)
    st.markdown("**Optional OI CSV upload**")
    st.caption("Columns: Symbol, OI, Prev_OI. Compare the same futures contract and expiry.")
    oi_file=st.file_uploader("Upload OI CSV",type=["csv"],key="oi")
    if oi_file:
        oi=pd.read_csv(oi_file)
        if {"Symbol","OI","Prev_OI"}.issubset(oi.columns):
            oi["OI Change %"]=(oi.OI/oi.Prev_OI.replace(0,np.nan)-1)*100
            if results:
                prices=pd.DataFrame([{"Symbol":r["Symbol"],"Change %":r["Change %"]} for r in results])
                oi=oi.merge(prices,on="Symbol",how="left")
            else: oi["Change %"]=np.nan
            oi["Interpretation"]=np.select([(oi["Change %"]>0)&(oi["OI Change %"]>0),(oi["Change %"]<0)&(oi["OI Change %"]>0),(oi["Change %"]>0)&(oi["OI Change %"]<0),(oi["Change %"]<0)&(oi["OI Change %"]<0)],["Long build-up","Short build-up","Short covering","Long unwinding"],default="Mixed / unavailable")
            st.dataframe(oi,use_container_width=True,hide_index=True)
        else: st.error("CSV must contain Symbol, OI, Prev_OI columns.")

with tabs[6]:
    st.subheader("Trading journal")
    st.warning("This app is public. Do not enter private notes or sensitive trading information until access protection is added.")
    with st.form("trade_form"):
        c1,c2,c3=st.columns(3)
        trade_date=c1.date_input("Trade date",date.today())
        symbol=c2.text_input("Symbol","RELIANCE").upper()
        side=c3.selectbox("Side",["LONG","SHORT"])
        c4,c5,c6=st.columns(3)
        setup=c4.selectbox("Setup",["Bullish OB","Bearish OB","Liquidity sweep","BOS retest","CHoCH","Other"])
        entry=c5.number_input("Entry",min_value=0.0,value=0.0,step=0.05)
        stop=c6.number_input("Stop loss",min_value=0.0,value=0.0,step=0.05)
        c7,c8,c9=st.columns(3)
        target=c7.number_input("Target",min_value=0.0,value=0.0,step=0.05)
        qty=c8.number_input("Quantity",min_value=1,value=1,step=1)
        exit_price=c9.number_input("Exit price (0 if open)",min_value=0.0,value=0.0,step=0.05)
        notes=st.text_area("Notes")
        submitted=st.form_submit_button("Save trade")
        if submitted:
            with sqlite3.connect(DB_PATH) as con:
                con.execute("INSERT INTO trades(trade_date,symbol,side,setup,entry,stop,target,exit_price,quantity,notes,status) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (str(trade_date),symbol,side,setup,entry,stop,target,exit_price,int(qty),notes,"Closed" if exit_price>0 else "Open"))
            st.success("Trade saved.")
    with sqlite3.connect(DB_PATH) as con:
        journal=pd.read_sql_query("SELECT * FROM trades ORDER BY trade_date DESC,id DESC",con)
    if not journal.empty:
        journal["P&L"]=np.where(journal.exit_price>0,np.where(journal.side=="LONG",(journal.exit_price-journal.entry)*journal.quantity,(journal.entry-journal.exit_price)*journal.quantity),np.nan)
        st.dataframe(journal,use_container_width=True,hide_index=True)
        st.download_button("Download journal CSV",journal.to_csv(index=False).encode(),"trading_journal.csv","text/csv")
    else: st.info("No trades recorded yet.")

with tabs[7]:
    st.subheader("Methodology, limitations and safety")
    st.markdown("""
- **Universe:** starter sample in `symbols.csv`, not guaranteed to be the complete current NSE F&O list.
- **Price feed:** Yahoo Finance via yfinance is unofficial; data may be delayed, incomplete, rate-limited or unavailable.
- **Trend:** EMA20 / EMA50 alignment.
- **Swing structure:** confirmed 3-left / 3-right pivots.
- **BOS / CHoCH:** close beyond a recent confirmed swing; simplified approximation.
- **Liquidity sweep:** wick beyond a recent swing and close back inside.
- **FVG retest:** three-candle gap followed by a later revisit that remains held at the latest close. This is a simplified rule.
- **4H:** resampled from hourly Yahoo candles, not exchange-native 4H data.
- **Risk levels:** only shown for a potential setup; based on 1.5 ATR stop and 2/3 ATR targets. Illustrative, not recommendations.
- **OI:** upload a correctly matched futures OI CSV; no dependable live OI feed is included.
- **No auto-trading:** the app does not connect to a broker or place orders.
""")
    st.markdown("NSE official derivatives reports: https://www.nseindia.com/all-reports-derivatives")
    st.warning("Never trade solely on scanner output. Check data quality, expiry, liquidity, slippage, position size and risk. Backtest and paper-trade first.")

import sqlite3
from datetime import datetime, date
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
        pd.DataFrame(DEFAULT_SYMBOLS, columns=["symbol","sector"]).to_csv(p, index=False)
    return pd.read_csv(p)

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trade_date TEXT, symbol TEXT,
            side TEXT, setup TEXT, entry REAL, stop REAL, target REAL,
            exit_price REAL, quantity INTEGER, notes TEXT, status TEXT DEFAULT 'Open'
        )""")
init_db()

def fetch_history(symbol, period, interval):
    ticker = symbol.strip().upper() + ".NS"
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                     progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    needed = ["Open","High","Low","Close","Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].copy().dropna()
    df.index = pd.to_datetime(df.index)
    return df

def indicators(df):
    x = df.copy()
    x["ATR"] = pd.concat([
        x["High"]-x["Low"],
        (x["High"]-x["Close"].shift()).abs(),
        (x["Low"]-x["Close"].shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    x["VolAvg20"] = x["Volume"].rolling(20).mean()
    x["AvgDailyMove"] = ((x["High"]-x["Low"])/x["Close"].shift()).rolling(50).mean()
    x["Return20"] = x["Close"].pct_change(20)
    return x

def analyse(symbol, sector, raw):
    if raw.empty or len(raw) < 55:
        return None
    x = indicators(raw)
    last = x.iloc[-1]
    prev = x.iloc[-2]
    close = float(last.Close)
    atr = float(last.ATR) if pd.notna(last.ATR) else close * .02
    if not np.isfinite(atr) or atr <= 0:
        atr = close * .02
    avg_range = float(last.AvgDailyMove) if pd.notna(last.AvgDailyMove) else .02
    strength = abs(close / float(x["Close"].iloc[-2]) - 1) / max(avg_range, .005)
    vol_ratio = float(last.Volume / last.VolAvg20) if pd.notna(last.VolAvg20) and last.VolAvg20 else np.nan
    trend_up = close > float(last.EMA20) > float(last.EMA50)
    trend_dn = close < float(last.EMA20) < float(last.EMA50)
    swing_high = float(x["High"].iloc[-21:-1].max())
    swing_low = float(x["Low"].iloc[-21:-1].min())
    bos_up = close > swing_high
    bos_dn = close < swing_low
    recent = x.iloc[-12:-1]
    # Liquidity sweep: trades beyond prior 10-bar extreme, then closes back inside.
    prior_hi = float(x["High"].iloc[-11:-1].max())
    prior_lo = float(x["Low"].iloc[-11:-1].min())
    sweep_high = float(last.High) > prior_hi and close < prior_hi
    sweep_low = float(last.Low) < prior_lo and close > prior_lo
    # Simple OB: most recent opposite candle in preceding 10 bars before a BOS.
    bull_ob = bear_ob = None
    if bos_up:
        for i in range(len(x)-2, max(-1, len(x)-13), -1):
            row = x.iloc[i]
            if row.Close < row.Open:
                bull_ob = (float(row.Low), float(row.High), x.index[i])
                break
    if bos_dn:
        for i in range(len(x)-2, max(-1, len(x)-13), -1):
            row = x.iloc[i]
            if row.Close > row.Open:
                bear_ob = (float(row.Low), float(row.High), x.index[i])
                break
    score = 0
    score += 2 if trend_up else (-2 if trend_dn else 0)
    score += 2 if bos_up else (-2 if bos_dn else 0)
    score += 1 if close > float(last.EMA20) else (-1 if close < float(last.EMA20) else 0)
    score += 1 if np.isfinite(vol_ratio) and vol_ratio >= 1.5 else 0
    score += 1 if sweep_low else (-1 if sweep_high else 0)
    direction = "Bullish" if score >= 3 else ("Bearish" if score <= -3 else "Neutral")
    side = "LONG" if direction == "Bullish" else "SHORT"
    entry = close
    stop = close - 1.5*atr if side == "LONG" else close + 1.5*atr
    target = close + 3*atr if side == "LONG" else close - 3*atr
    return {
        "Symbol":symbol, "Sector":sector, "Close":round(close,2), "Change %":round((close/float(prev.Close)-1)*100,2),
        "Score":score, "Bias":direction, "Strength":round(float(strength),2),
        "Volume x":round(vol_ratio,2) if np.isfinite(vol_ratio) else np.nan,
        "BOS":"Bullish" if bos_up else ("Bearish" if bos_dn else "—"),
        "Liquidity sweep":"Sell-side" if sweep_low else ("Buy-side" if sweep_high else "—"),
        "Entry":round(entry,2), "Stop":round(stop,2), "Target":round(target,2),
        "ATR":round(atr,2), "Bull OB":bull_ob, "Bear OB":bear_ob, "data":x
    }

def make_chart(result):
    x = result["data"].tail(100)
    fig = go.Figure(data=[go.Candlestick(x=x.index, open=x.Open, high=x.High, low=x.Low, close=x.Close, name="Price")])
    fig.add_trace(go.Scatter(x=x.index, y=x.EMA20, name="EMA 20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x.index, y=x.EMA50, name="EMA 50", line=dict(width=1)))
    for key, color, label in [("Bull OB","rgba(0,180,80,0.20)","Bullish OB"),("Bear OB","rgba(220,60,60,0.20)","Bearish OB")]:
        ob=result[key]
        if ob:
            lo, hi, stamp=ob
            fig.add_hrect(y0=lo,y1=hi,fillcolor=color,line_width=0,annotation_text=label,annotation_position="top left")
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
    return fig

symbols_df = ensure_symbols()
st.sidebar.header("Scanner settings")
interval_label = st.sidebar.selectbox("Candle timeframe", ["1 day", "15 minutes", "5 minutes"], index=0)
interval = {"1 day":"1d","15 minutes":"15m","5 minutes":"5m"}[interval_label]
period = {"1d":"1y","15m":"60d","5m":"60d"}[interval]
max_symbols = st.sidebar.slider("Number of stocks to scan", 5, min(60, len(symbols_df)), min(25,len(symbols_df)))
st.sidebar.caption("Starter list is editable in symbols.csv. Refresh the F&O list from NSE's official contract information page.")
scan_button = st.sidebar.button("🔄 Run / refresh scan", type="primary")

if "results" not in st.session_state:
    st.session_state.results = []
if scan_button:
    out=[]
    progress=st.progress(0)
    status=st.empty()
    subset=symbols_df.head(max_symbols)
    for i, row in enumerate(subset.itertuples(index=False), start=1):
        status.write(f"Fetching {row.symbol} ({i}/{len(subset)})…")
        try:
            hist=fetch_history(row.symbol,period,interval)
            res=analyse(row.symbol,row.sector,hist)
            if res: out.append(res)
        except Exception:
            pass
        progress.progress(i/len(subset))
    st.session_state.results=out
    status.write(f"Scan complete: {len(out)} stocks returned data.")
results=st.session_state.results

tabs=st.tabs(["Scanner","Order blocks & chart","Sector & breadth","OI / volume","Trading journal","How it works"])
with tabs[0]:
    st.subheader("Bullish and bearish stock scanner")
    if results:
        table=pd.DataFrame([{k:v for k,v in r.items() if k not in ["data","Bull OB","Bear OB"]} for r in results])
        c1,c2,c3=st.columns(3)
        c1.metric("Bullish",int((table.Bias=="Bullish").sum()))
        c2.metric("Bearish",int((table.Bias=="Bearish").sum()))
        c3.metric("Neutral",int((table.Bias=="Neutral").sum()))
        filt=st.selectbox("Show",["All","Bullish","Bearish","Neutral"])
        view=table if filt=="All" else table[table.Bias==filt]
        st.dataframe(view.sort_values(["Score","Strength"],ascending=False),use_container_width=True,hide_index=True)
        st.download_button("Download scanner CSV",table.to_csv(index=False).encode(), "scanner_results.csv","text/csv")
    else:
        st.info("Choose settings on the left and click Run / refresh scan. First run downloads public Yahoo Finance data.")
with tabs[1]:
    st.subheader("Order-block chart")
    if results:
        names=[r["Symbol"] for r in results]
        chosen=st.selectbox("Choose stock",names)
        r=next(z for z in results if z["Symbol"]==chosen)
        st.plotly_chart(make_chart(r),use_container_width=True)
        a,b,c=st.columns(3)
        a.metric("Bias",r["Bias"])
        b.metric("BOS",r["BOS"])
        c.metric("Liquidity sweep",r["Liquidity sweep"])
        st.write("Bullish OB:", r["Bull OB"][:2] if r["Bull OB"] else "Not detected")
        st.write("Bearish OB:", r["Bear OB"][:2] if r["Bear OB"] else "Not detected")
        st.caption("OB logic is a simple, rule-based approximation: last opposite candle before a 20-bar closing break. It is not a universal institutional definition.")
    else: st.info("Run a scan first.")
with tabs[2]:
    st.subheader("Sector strength & market breadth")
    if results:
        t=pd.DataFrame([{k:v for k,v in r.items() if k not in ["data","Bull OB","Bear OB"]} for r in results])
        sector=t.groupby("Sector").agg(Stocks=("Symbol","count"),AvgChange=("Change %","mean"),Bullish=("Bias",lambda s:(s=="Bullish").sum()),Bearish=("Bias",lambda s:(s=="Bearish").sum())).reset_index()
        sector["AvgChange"]=sector["AvgChange"].round(2)
        st.dataframe(sector.sort_values("AvgChange",ascending=False),use_container_width=True,hide_index=True)
        st.metric("Market breadth (bullish minus bearish)",int((t.Bias=="Bullish").sum()-(t.Bias=="Bearish").sum()))
    else: st.info("Run a scan first.")
with tabs[3]:
    st.subheader("Open interest & volume")
    st.write("Volume expansion is calculated from price history. Yahoo Finance does not provide dependable NSE stock-futures OI history through this free feed.")
    if results:
        t=pd.DataFrame([{k:v for k,v in r.items() if k not in ["data","Bull OB","Bear OB"]} for r in results])
        st.dataframe(t[["Symbol","Volume x","Change %","Bias"]].sort_values("Volume x",ascending=False),use_container_width=True,hide_index=True)
    st.markdown("**Optional OI CSV upload**")
    st.caption("Prepare a CSV with columns: Symbol, OI, Prev_OI. Values should refer to the same futures contract/expiry on consecutive snapshots or sessions.")
    oi_file=st.file_uploader("Upload OI CSV",type=["csv"],key="oi")
    if oi_file:
        oi=pd.read_csv(oi_file)
        required={"Symbol","OI","Prev_OI"}
        if required.issubset(oi.columns):
            oi["OI Change %"]=(oi.OI/oi.Prev_OI.replace(0,np.nan)-1)*100
            if results:
                prices=pd.DataFrame([{"Symbol":r["Symbol"],"Change %":r["Change %"]} for r in results])
                oi=oi.merge(prices,on="Symbol",how="left")
            oi["Interpretation"]=np.select([(oi["Change %"]>0)&(oi["OI Change %"]>0),(oi["Change %"]<0)&(oi["OI Change %"]>0),(oi["Change %"]>0)&(oi["OI Change %"]<0),(oi["Change %"]<0)&(oi["OI Change %"]<0)],["Long build-up","Short build-up","Short covering","Long unwinding"],default="Mixed / unavailable")
            st.dataframe(oi,use_container_width=True,hide_index=True)
        else: st.error("CSV must contain Symbol, OI, Prev_OI columns.")
with tabs[4]:
    st.subheader("Trading journal")
    with st.form("trade_form"):
        c1,c2,c3=st.columns(3)
        trade_date=c1.date_input("Trade date",date.today())
        symbol=c2.text_input("Symbol", "RELIANCE").upper()
        side=c3.selectbox("Side",["LONG","SHORT"])
        c4,c5,c6=st.columns(3)
        setup=c4.selectbox("Setup",["Bullish OB","Bearish OB","Liquidity sweep","BOS retest","Other"])
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
        journal["P&L"] = np.where(journal.exit_price>0,
            np.where(journal.side=="LONG",(journal.exit_price-journal.entry)*journal.quantity,(journal.entry-journal.exit_price)*journal.quantity),np.nan)
        st.dataframe(journal,use_container_width=True,hide_index=True)
        st.download_button("Download journal CSV",journal.to_csv(index=False).encode(),"trading_journal.csv","text/csv")
    else: st.info("No trades recorded yet.")
with tabs[5]:
    st.subheader("Methodology, limits and safety")
    st.markdown("""
- **Universe:** starter sample of NSE F&O underlyings in `symbols.csv`. Update it using NSE's official F&O contract information / permitted underlying list.
- **Price feed:** `yfinance` accesses Yahoo Finance public endpoints. It is unofficial, can be delayed, rate-limited, incomplete or unavailable. It is not an exchange-authorised live feed.
- **Trend:** EMA20 / EMA50 alignment.
- **BOS:** close above/below the previous 20-bar high/low.
- **Liquidity sweep:** candle breaches the previous 10-bar extreme and closes back inside.
- **Order block:** most recent opposite-colour candle in the 10 bars before the BOS.
- **Strength:** absolute latest one-bar return divided by average daily high-low percentage over the prior 50 bars. For intraday scans this remains a rough proxy, not OrderBlock's proprietary score.
- **Risk levels:** entry at latest close; stop at 1.5 ATR; target at 3 ATR. These are illustrative levels, not signals or guaranteed outcomes.
- **OI:** import a properly matched futures OI CSV. Do not mix expiries or compare unlike contracts.
- **Sector breadth:** only the stocks in your currently scanned sample; not the entire NSE market.
- **No auto-trading:** this app never places orders or connects to your broker.
""")
    st.markdown("NSE official F&O reports and contract information: https://www.nseindia.com/all-reports-derivatives")
    st.warning("Do not trade solely on scanner output. Validate data, contract expiry, liquidity, slippage, corporate actions and position sizing. Paper-trade and backtest rules first.")

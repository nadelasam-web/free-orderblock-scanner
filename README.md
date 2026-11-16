# Free Order-Block Scanner (NSE F&O starter)

A beginner-friendly Streamlit research dashboard. It is a practical approximation, not a clone of OrderBlock.com and not investment advice.

## Features
- Bullish / bearish / neutral scanner with trend, BOS, volume and liquidity-sweep checks
- Rule-based bullish/bearish order-block zones and candlestick chart
- Sector strength and sample-universe breadth
- Volume expansion and optional OI CSV import
- Entry / stop / target estimates using ATR
- Local SQLite trading journal and CSV export

## Important data limitations
The default price feed is `yfinance` / Yahoo Finance public data. It is unofficial and may be delayed, throttled, incomplete or unavailable. Free Yahoo Finance data is not a dependable real-time NSE feed and does not provide dependable stock-futures OI history. OI is therefore an optional CSV import, not fabricated.

The starter `symbols.csv` contains a sample of liquid F&O names, not a guaranteed current complete F&O universe. Refresh it from the official NSE derivatives contract information page:
https://www.nseindia.com/static/products-services/equity-derivatives-contract-information
The current eligible universe changes. Use the current permitted-underlying list and verify symbols before relying on the scanner.

## Run on Windows
1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Extract this ZIP.
3. Open Command Prompt in the extracted folder.
4. Run:
   ```bash
   py -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. A browser tab opens at the local dashboard address. On the sidebar choose timeframe and scan size, then click **Run / refresh scan**.

## Run on Mac / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Use on Android
The easiest no-PC route is Streamlit Community Cloud:
1. Create a GitHub account and a public repository.
2. Upload `app.py`, `requirements.txt`, and `symbols.csv` to the repository.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select the repository, branch, and `app.py`, then deploy.
5. Open the app URL in Chrome on your phone.

Community Cloud is free but resources / sleep behaviour can change. Your SQLite journal on a hosted app may not be persistent across rebuilds/restarts. For a durable journal, download CSV backups regularly or use a persistent database.

## Symbols list
Edit `symbols.csv` with two columns:
```csv
symbol,sector
RELIANCE,Energy
HDFCBANK,Banking
```
Use Yahoo Finance NSE suffix internally; the app adds `.NS`.

## OI CSV format
Upload a CSV with:
```csv
Symbol,OI,Prev_OI
RELIANCE,123456,118000
```
Make sure OI and Prev_OI are for the same underlying futures contract and comparable expiry/session. The simple interpretation combines underlying price change with OI change.

## How the strategy rules work
- Trend: EMA20 / EMA50 alignment
- BOS: close breaks previous 20-bar high or low
- Liquidity sweep: wick breaches previous 10-bar extreme and closes back inside
- OB: last opposite-colour candle in 10 bars before BOS
- Risk levels: close ± 1.5 ATR stop and ± 3 ATR target

These are deliberately transparent starter rules. They are not a universal institutional definition and do not reproduce proprietary OrderBlock algorithms. Test them on historical data and paper trade before risking money.

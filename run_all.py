# run_all.py

import os
import pandas as pd
import yaml
import asyncio
from telegram import Bot

from data_ingest import fetch_data
from strategy import generate_signals, backtest_strategy
from sheet_logger import log_trades, update_summary 

# Load Telegram config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

bot = Bot(token=cfg["telegram"]["token"])
CHAT_ID = cfg["telegram"]["chat_id"]

def send_alert(message: str):
    """
    Synchronously send a Telegram message by running the async coroutine.
    """
    async def _send():
        await bot.send_message(chat_id=CHAT_ID, text=message)
    try:
        asyncio.run(_send())
        print("✅ Telegram alert sent.")
    except Exception as e:
        print("❌ Telegram error:", e)

def load_and_clean(csv_path):
    # Cleaning up the csv's
    df = pd.read_csv(csv_path)
    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Close','Volume'], inplace=True)
    return df

def main(): # main pipeline
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    all_trades = []

    print("📥 Fetching fresh data...")
    fetch_data(tickers, period="6mo", interval="1d", save_csv=True)

    for ticker in tickers:
        csv_path = os.path.join("data", ticker.replace(".", "_") + ".csv")
        if not os.path.exists(csv_path):
            continue

        print(f"\n📊 Running strategy for {ticker}")
        df = load_and_clean(csv_path)
        df = generate_signals(df)
        trades, _ = backtest_strategy(df) # backtesting on the historical data

        # Alert on the most recent buy signal
        if df.iloc[-1]["Signal"] == 1:
            price = df.iloc[-1]["Close"]
            send_alert(f"📈 BUY signal for {ticker} at ₹{price:.2f}")

        all_trades.extend(trades)

    print("\n📝 Logging trades to Google Sheets...")
    if all_trades:
        log_trades(all_trades)
        update_summary()
    else:
        print("No trades to log today.")

    total_pl = sum(t.get("P&L", 0) for t in all_trades)
    send_alert(f"📊 Strategy run complete. Net P&L: ₹{total_pl:.2f}")

if __name__ == "__main__":
    main()

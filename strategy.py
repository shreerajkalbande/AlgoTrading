# strategy.py

import pandas as pd
import numpy as np
import os

def compute_indicators(df):
    """Add RSI(14), SMA20, SMA50 to DataFrame."""
    delta = df['Close'].diff()

    # Calculate average gains/losses
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    return df

def generate_signals(df):
    """Set Signal=1 when RSI<30 AND SMA20 > SMA50."""
    df = compute_indicators(df)
    df['Signal'] = 0
    mask = (df['RSI'] < 30) & (df['SMA20'] > df['SMA50'])
    df.loc[mask, 'Signal'] = 1
    return df

def backtest_strategy(df, initial_cash=100000):
    """
    Simulate buy/sell:
      - Buy on Signal==1 if flat.
      - Sell when RSI>70 or SMA20<SMA50.
    """
    cash = initial_cash
    position = 0
    trade_log = []
    last_buy_price = 0

    for idx, row in df.iterrows():
        if row['Signal'] == 1 and position == 0:
            # BUY
            price = row['Close']
            shares = cash // price
            if shares > 0:
                cost = shares * price
                cash -= cost
                position = shares
                last_buy_price = price
                trade_log.append({
                    'Date': idx,
                    'Action': 'BUY',
                    'Price': price,
                    'Shares': shares
                })

        # SELL condition: either RSI>70 or MA cross down
        elif position > 0 and (row['RSI'] > 70 or row['SMA20'] < row['SMA50']):
            price = row['Close']
            proceeds = position * price
            pnl = position * (price - last_buy_price)
            cash += proceeds
            trade_log.append({
                'Date': idx,
                'Action': 'SELL',
                'Price': price,
                'Shares': position,
                'P&L': pnl
            })
            position = 0

    net_profit = cash - initial_cash
    return trade_log, net_profit

def run_strategy_on_csv(csv_path):
    # 1) Load, drop bad first row, parse dates
    df = pd.read_csv(csv_path)

    # Drop any rows where Date is missing or not parseable
    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # 2) Ensure numeric types
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # 3) Generate signals + backtest
    df_signals = generate_signals(df)
    trades, profit = backtest_strategy(df_signals)

    return trades, profit

if __name__ == "__main__":
    data_folder = "data"
    total_profit = 0.0

    for fname in os.listdir(data_folder):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(data_folder, fname)
        print(f"\n=== {fname} ===")
        trades, profit = run_strategy_on_csv(path)
        for t in trades:
            date = t['Date'].strftime('%Y-%m-%d')
            if t['Action'] == 'BUY':
                print(f"{date} BUY  {t['Shares']} @ ₹{t['Price']:.2f}")
            else:
                print(f"{date} SELL {t['Shares']} @ ₹{t['Price']:.2f} → P&L: ₹{t['P&L']:.2f}")
        print(f"Net Profit for {fname.split('_')[0]}: ₹{profit:.2f}")
        total_profit += profit

    print(f"\n>>> TOTAL PORTFOLIO PROFIT: ₹{total_profit:.2f}")

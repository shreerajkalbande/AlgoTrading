# data_ingest.py

import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, period="6mo", interval="1d", save_csv=True, output_dir="data"):
    """
    Fetches historical stock data using yfinance.

    Args:
        tickers (list): List of stock symbols (e.g. ["RELIANCE.NS", "TCS.NS"]).
        period (str): Period of data to fetch (e.g. "6mo", "1y").
        interval (str): Data interval (e.g. "1d" for daily).
        save_csv (bool): Whether to save the data as CSV files.
        output_dir (str): Directory to store CSVs.

    Returns:
        dict: Dictionary of DataFrames keyed by ticker.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dict = {}

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, period=period, interval=interval)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df["Ticker"] = ticker
        data_dict[ticker] = df

        if save_csv:
            file_path = os.path.join(output_dir, f"{ticker.replace('.','_')}.csv")
            df.to_csv(file_path, index=False)
            print(f"Saved: {file_path}")

    return data_dict

# Example usage
if __name__ == "__main__":
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    fetch_data(tickers)

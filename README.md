###  Algo-Trading System with ML, Automation, and Google Sheets Integration

I made this automated pipeline for backtesting the simple trading strategy and analysis of ML models only.
Created by: Shreeraj Kalbande

---
This is a fully automated Python-based **algorithmic trading prototype** that:

- 📊 Fetches stock market data using `yfinance`
- 📉 Implements a rule-based trading strategy (RSI + DMA crossover)
- 🧠 Includes machine learning models to predict next-day price movements
- 📤 Logs trades and performance to **Google Sheets** automatically
- 📲 Sends real-time **Telegram alerts** for trade signals and summaries
- 🔄 Is modular, reproducible, and extensible for future development

## Features & Modules

### 1. Data Ingestion

- Uses "yfinance" to fetch **daily** price data for:
  - RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`
- Stores data in `/data` folder as CSVs

### 2. Trading Strategy

Rule-Based Logic:
- Buy when RSI < 30
- Confirm if 20-day DMA > 50-day DMA (bullish trend)
- Sell when RSI > 70 or price rises 3%

### 3. Backtesting

- Runs strategy for each stock
- Simulates trade entry/exit
- Calculates net profit/loss (P&L) over last 6 months

### 4. ML Automation

Predicts next-day movement (Up/Down) using:
- RSI, MACD, On-Balance Volume, VWAP, Volume stats, Price return
- Models:
  - Logistic Regression (balanced)
  - Decision Tree + GridSearch hyperparameter tuning
- Reports accuracy, F1, recall, precision

### 5. Google Sheets Integration

- Automatically writes:
  - Trade log
  - Summary P&L (win/loss ratio, total profit)
- Uses Google Sheets API with service account credentials

### 6. Telegram Alerts

- Sends a Telegram message after full pipeline run
  - Reports net P&L and stocks traded

---

## Setup Instructions

### 1. Environment Setup

```bash
conda create -n trading_env python=3.11
conda activate trading_env
pip install -r requirements.txt
```
### 2. Run
# Create these 2 files first for Successful Run

service_account.json -
    This file contains the OAuth2 credentials required to authenticate your app with Google Sheets and Google Drive APIs. It’s automatically         generated when you create a Google Service Account in Google Cloud Console.
    Create service_account.json that links the Google Sheets to this as well as config.yaml for linking telegram to this 
  - Purpose:
    Authorizes the script to read/write Google Sheets on your behalf.
    Allows automated logging of trades, profit/loss summaries, and strategy outputs.

config.yaml -
    This YAML file contains your Telegram Bot API token and Chat ID for pushing trade alerts or error notifications via Telegram.
  - Purpose:
    Authenticates requests to the Telegram Bot API
    Sends real-time updates about your trading strategy execution (e.g., Net P&L) directly to your Telegram inbox

```bash
python run_all.py
```

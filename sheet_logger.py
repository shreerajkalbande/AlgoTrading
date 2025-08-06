# sheet_logger.py

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Constants
CREDS_FILE = "service_account.json"
SPREADSHEET_NAME = "AlgoTradeLog"  # your sheet name

# Scopes for Google Sheets API
SCOPES = ["https://spreadsheets.google.com/feeds",
          "https://www.googleapis.com/auth/drive"]

def auth_sheets():
    """Authenticate and return a gspread client."""
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPES)
    client = gspread.authorize(creds)
    return client

def init_sheets():
    """Open spreadsheet and return the three worksheets."""
    client = auth_sheets()
    sh = client.open(SPREADSHEET_NAME)
    ws_trades = sh.worksheet("Trade Log")
    ws_pl     = sh.worksheet("P&L Summary")
    ws_wr     = sh.worksheet("Win Ratio")
    return ws_trades, ws_pl, ws_wr

def log_trades(trades):
    """
    Append trades to the 'Trade Log' sheet.
    trades: list of dicts with keys ['Date','Action','Price','Shares','P&L'].
    """
    ws_trades, _, _ = init_sheets()
    for t in trades:
        # Some BUY entries may lack P&L field
        pnl = t.get("P&L", "")
        row = [t["Date"].strftime("%Y-%m-%d"), t["Action"],
               t["Shares"], round(t["Price"],2), pnl]
        ws_trades.append_row(row, value_input_option="USER_ENTERED")

def update_summary():
    """
    Read the Trade Log and update P&L Summary & Win Ratio tabs.
    """
    ws_trades, ws_pl, ws_wr = init_sheets()
    # Fetch all log data
    data = ws_trades.get_all_records(empty2zero=True)
    df = pd.DataFrame(data)

    # If no trades yet, exit
    if df.empty:
        return

    # Compute summary P&L per ticker
    # Assume ticker column in log? If not, use one global summary.
    df['P&L'] = pd.to_numeric(df['P&L'], errors='coerce')
    total_pl = df['P&L'].sum()
    wins = df[df['P&L'] > 0].shape[0]
    losses = df[df['P&L'] < 0].shape[0]
    win_ratio = wins / (wins + losses) if (wins+losses) > 0 else 0

    # Write P&L Summary
    ws_pl.clear()
    ws_pl.append_row(["Total P&L", "Wins", "Losses"])
    ws_pl.append_row([round(total_pl,2), wins, losses])

    # Write Win Ratio
    ws_wr.clear()
    ws_wr.append_row(["Win Ratio"])
    ws_wr.append_row([round(win_ratio,4)])  # e.g. 0.75 for 75%

if __name__ == "__main__":
    # Example: read trades from previous run
    from strategy import run_strategy_on_csv
    import os

    all_trades = []
    for fname in os.listdir("data"):
        if fname.endswith(".csv"):
            trades, _ = run_strategy_on_csv(os.path.join("data", fname))
            all_trades.extend(trades)

    # Log and update
    log_trades(all_trades)
    update_summary()

    print("✅ Google Sheets updated with trades and summary.")

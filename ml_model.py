'''
Low Accuracy score reasons :
1. Financial markets are inherently noisy
2. Only technical indicators used (no macro/news/trends)
3. Small dataset
'''
"The goal was feature exploration and pipeline automation"

"This model can be improved by adding news sentiment, fundamentals, or price action patterns"


import os
import warnings

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

# Suppress warnings about undefined metrics and future changes
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = "data"

# Final feature list including extra volume features
FEATURES = [
    "RSI",
    "MACD",
    "MACD_Signal",
    "Return",
    "Vol_Change",
    "Vol_SMA14",
    "Vol_Spike",
    "Vol_PctRank",
    "OBV",
    "AccDist",
    "VWAP"
]

def load_and_engineer():
    """Load all CSVs, engineer the full set of features, and return a cleaned DataFrame."""
    frames = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        # --- Clean & index by date ---
        df = df[df['Date'].notna()].copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        # Coerce numeric columns
        for col in ['Open','High','Low','Close','Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Close','Volume'], inplace=True)

        # --- Technical features ---
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['Return'] = df['Close'].pct_change()

        # --- Volume features ---
        df['Vol_Change'] = df['Volume'].pct_change()
        df['Vol_SMA14'] = df['Volume'].rolling(window=14).mean()
        df['Vol_Spike'] = (df['Volume'] > 2 * df['Vol_SMA14']).astype(int)
        df['Vol_PctRank'] = (
            df['Volume']
            .rolling(window=20)
            .apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
        )
        df['OBV'] = OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume']
        ).on_balance_volume()
        df['AccDist'] = (df['Close'] - df['Open']) * df['Volume']
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume'],
            window=14
        ).volume_weighted_average_price()

        # --- Label for next-day movement ---
        df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        frames.append(df)

    # Combine all, drop rows missing any feature or label
    data = pd.concat(frames)
    data.dropna(subset=FEATURES + ['Label'], inplace=True)
    return data

def evaluate_cross_validation(X, y):
    """Perform 5-fold CV and print mean/std accuracy for LR and DT."""
    print("\n--- 5-Fold Cross-Validation Scores ---")
    lr = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        solver='liblinear'
    )
    dt = DecisionTreeClassifier(
        class_weight='balanced'
    )
    for name, model in [('LogisticRegression', lr), ('DecisionTree', dt)]:
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

def tune_decision_tree(X, y):
    """Grid-search DT max_depth over [3,5,7,9] and return best estimator."""
    print("\n--- Decision Tree Hyperparameter Tuning ---")
    param_grid = {'max_depth': [3, 5, 7, 9]}
    dt = DecisionTreeClassifier(class_weight='balanced')
    grid = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    best = grid.best_estimator_
    print(f"Best max_depth: {grid.best_params_['max_depth']}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return best

def train_and_report(X_train, X_test, y_train, y_test, tuned_dt):
    """Train LR & default DT, evaluate on test; then evaluate tuned DT."""
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=2000, class_weight='balanced', solver='liblinear'
        ),
        'DecisionTree': DecisionTreeClassifier(
            class_weight='balanced', max_depth=5
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {name} (Test Set) ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(
            y_test, preds, target_names=['Down', 'Up'],
            digits=4, zero_division=0
        ))

    # Evaluate tuned DT
    preds = tuned_dt.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== Tuned DecisionTree (Test Set) ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(
        y_test, preds, target_names=['Down', 'Up'],
        digits=4, zero_division=0
    ))

if __name__ == "__main__":
    print("Loading and engineering features…")
    df = load_and_engineer()
    print(f"Total samples: {len(df)}")

    X = df[FEATURES]
    y = df['Label']

    # Train-test split (preserve time order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 1) Cross-validation
    evaluate_cross_validation(X_train, y_train)

    # 2) Hyperparameter tuning
    tuned_dt = tune_decision_tree(X_train, y_train)

    # 3) Final evaluation on test set
    train_and_report(X_train, X_test, y_train, y_test, tuned_dt)

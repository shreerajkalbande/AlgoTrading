"""
Quantitative ML Model for Price Direction Prediction

This module implements machine learning models for predicting next-day price movements
using technical indicators and market microstructure features.

Note: Financial markets are inherently noisy. Model performance is limited by:
1. Market efficiency and random walk properties
2. Limited feature set (technical indicators only)
3. Regime changes and non-stationarity

Enhancements for production:
- Alternative data (sentiment, news, fundamentals)
- Ensemble methods and feature engineering
- Regime detection and adaptive models
"""

import os
import warnings
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from config.settings import DATA_DIR, MODELS_DIR, ML_CONFIG

warnings.filterwarnings("ignore")

# Professional quantitative feature set
FEATURES = [
    # Momentum indicators
    "RSI", "Stoch_K", "Stoch_D", "CCI", "ROC_10", "ROC_20",
    # Trend indicators
    "ADX", "EMA_12", "EMA_26", "SMA_20", "SMA_50", "Hull_MA",
    # Volatility indicators
    "ATR", "BB_Upper", "BB_Lower", "BB_Width", "DC_Upper", "DC_Lower",
    # Volume/Flow indicators
    "VWAP", "OBV", "CMF", "Vol_Ratio", "Price_Volume_Trend",
    # Statistical/Mean reversion
    "Z_Score_20", "Price_Zscore", "Volume_Zscore", "Return_Zscore",
    # Price action
    "Return_1d", "Return_5d", "Volatility_20d", "Price_Position"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer comprehensive technical and quantitative features."""
    # Momentum indicators
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # CCI (Commodity Channel Index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Rate of Change
    df['ROC_10'] = df['Close'].pct_change(10) * 100
    df['ROC_20'] = df['Close'].pct_change(20) * 100
    
    # ADX for trend strength
    df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    
    # Trend indicators
    df['EMA_12'] = EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(df['Close'], window=26).ema_indicator()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Hull Moving Average
    wma_half = df['Close'].rolling(9).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)))
    wma_full = df['Close'].rolling(18).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)))
    hull_raw = 2 * wma_half - wma_full
    df['Hull_MA'] = hull_raw.rolling(4).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)))
    
    # Volatility indicators
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    bb = BollingerBands(df['Close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
    
    # Donchian Channels
    df['DC_Upper'] = df['High'].rolling(20).max()
    df['DC_Lower'] = df['Low'].rolling(20).min()
    
    # Volume/Flow indicators
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['VWAP'] = VolumeWeightedAveragePrice(
        df['High'], df['Low'], df['Close'], df['Volume'], window=20
    ).volume_weighted_average_price()
    
    # Chaikin Money Flow
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    # Volume analysis
    vol_sma = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / vol_sma
    
    # Price Volume Trend
    df['Price_Volume_Trend'] = (df['Close'].pct_change() * df['Volume']).cumsum()
    
    # Statistical/Mean reversion indicators
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['Z_Score_20'] = (df['Close'] - sma_20) / std_20
    
    # Price Z-score (mean reversion signal)
    df['Price_Zscore'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).std()
    
    # Volume Z-score
    df['Volume_Zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    
    # Return Z-score
    returns = df['Close'].pct_change()
    df['Return_Zscore'] = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
    
    # Price action features
    df['Return_1d'] = returns
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
    df['Price_Position'] = (df['Close'] - df['Close'].rolling(20).min()) / (
        df['Close'].rolling(20).max() - df['Close'].rolling(20).min()
    )
    
    return df

def load_and_prepare_data() -> pd.DataFrame:
    """Load all stock data and prepare features for ML training."""
    frames = []
    
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue
            
        df = pd.read_csv(DATA_DIR / fname)
        df = df[df['Date'].notna()].copy()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        
        # Clean numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Close', 'Volume'], inplace=True)
        
        # Add ticker identifier
        df['Ticker'] = fname.replace('.csv', '').replace('_', '.')
        
        # Engineer features
        df = engineer_features(df)
        
        # Create target variable (next day return > 0)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        frames.append(df)
    
    # Combine all data
    data = pd.concat(frames, sort=False)
    
    # Remove rows with missing features or target
    data.dropna(subset=FEATURES + ['Target'], inplace=True)
    
    return data

def evaluate_models_cv(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple models using time series cross-validation."""
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=ML_CONFIG['random_state']
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=ML_CONFIG['random_state']
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, random_state=ML_CONFIG['random_state']
        )
    }
    
    # Use TimeSeriesSplit for proper time series validation
    tscv = TimeSeriesSplit(n_splits=ML_CONFIG['cv_folds'])
    results = {}
    
    print("\n=== Time Series Cross-Validation Results ===")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        results[name] = {
            'accuracy_mean': scores.mean(),
            'accuracy_std': scores.std()
        }
        print(f"{name:18s}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    return results

def select_features(X: pd.DataFrame, y: pd.Series, k: int = 15) -> List[str]:
    """Select top k features using statistical tests."""
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    selected_features = feature_scores.head(k)['feature'].tolist()
    print(f"\nTop {k} selected features:")
    for i, (_, row) in enumerate(feature_scores.head(k).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:15s} (score: {row['score']:.2f})")
    
    return selected_features

def train_final_models(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> Dict:
    """Train and evaluate final models with proper metrics."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=ML_CONFIG['random_state']
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced', 
            random_state=ML_CONFIG['random_state']
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=ML_CONFIG['random_state']
        )
    }
    
    results = {}
    print("\n=== Final Model Performance ===")
    
    for name, model in models.items():
        # Use scaled data for LogisticRegression, original for tree-based
        X_tr = X_train_scaled if 'Logistic' in name else X_train
        X_te = X_test_scaled if 'Logistic' in name else X_test
        
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'scaler': scaler if 'Logistic' in name else None,
            'accuracy': accuracy,
            'auc': auc_score
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC-ROC:  {auc_score:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up'], digits=4))
    
    return results

def save_model(model_results: Dict, selected_features: List[str]) -> None:
    """Save the best performing model and metadata."""
    # Find best model by AUC score
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc'])
    best_model_data = model_results[best_model_name]
    
    model_package = {
        'model': best_model_data['model'],
        'scaler': best_model_data['scaler'],
        'features': selected_features,
        'performance': {
            'accuracy': best_model_data['accuracy'],
            'auc': best_model_data['auc']
        },
        'model_type': best_model_name
    }
    
    model_path = MODELS_DIR / 'best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\nBest model ({best_model_name}) saved to {model_path}")
    print(f"Performance: Accuracy={best_model_data['accuracy']:.4f}, AUC={best_model_data['auc']:.4f}")

def main():
    """Main execution pipeline for ML model training and evaluation."""
    print("=== Quantitative ML Pipeline ===")
    print("Loading and preparing data...")
    
    # Load and prepare data
    data = load_and_prepare_data()
    print(f"Total samples: {len(data):,}")
    print(f"Features: {len(FEATURES)}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Check class balance
    class_dist = data['Target'].value_counts(normalize=True)
    print(f"Class distribution: Down={class_dist[0]:.3f}, Up={class_dist[1]:.3f}")
    
    # Feature selection
    X_all = data[FEATURES]
    y = data['Target']
    selected_features = select_features(X_all, y, k=15)
    X = X_all[selected_features]
    
    # Time-aware train-test split
    split_idx = int(len(data) * 0.8)
    split_date = data.index[split_idx]
    train_mask = data.index <= split_date
    
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    
    print(f"\nTrain samples: {len(X_train):,} (until {split_date.date()})")
    print(f"Test samples:  {len(X_test):,} (from {X_test.index.min().date()})")
    
    # Cross-validation evaluation
    cv_results = evaluate_models_cv(X_train, y_train)
    
    # Train final models
    model_results = train_final_models(X_train, X_test, y_train, y_test)
    
    # Save best model
    save_model(model_results, selected_features)
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()

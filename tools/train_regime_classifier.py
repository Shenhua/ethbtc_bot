#!/usr/bin/env python3
"""
Train Random Forest Classifier for ML Regime Detection.

This script trains a model to predict regime transitions BEFORE they happen,
using features from the regime_features module. Walk-forward cross-validation
is used to detect overfitting.

Usage:
    python tools/train_regime_classifier.py \
        --data data/raw/ETHBTC_15m_2021-2025_vision.csv \
        --funding data/raw/ETHUSDT_funding_2021-2025.csv \
        --fear-greed data/raw/fear_greed_index_2021-2025.csv \
        --output models/regime_classifier_v1.pkl

Labels:
    The model predicts whether the regime will be "Trend" (1) or "Mean Reversion" (0)
    in the NEXT 16 bars (4 hours). Labels are created by looking at the FUTURE ADX
    regime score, then shifted to avoid look-ahead bias during prediction.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Import our modules
from core.regime_features import build_regime_features, get_feature_names
from core.regime import get_regime_score
from core.ethbtc_accum_bot import load_vision_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("train_regime")


def load_data(
    data_path: str,
    funding_path: str = None,
    fear_greed_path: str = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load OHLCV, funding, and fear/greed data."""
    
    log.info(f"Loading OHLCV from {data_path}")
    df = load_vision_csv(data_path)
    log.info(f"  Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    
    funding = None
    if funding_path:
        log.info(f"Loading funding from {funding_path}")
        funding_df = pd.read_csv(funding_path, index_col=0, parse_dates=True)
        # Handle different column names
        if "fundingRate" in funding_df.columns:
            funding = funding_df["fundingRate"]
        elif "rate" in funding_df.columns:
            funding = funding_df["rate"]
        else:
            funding = funding_df.iloc[:, 0]
        # Ensure index is timezone-aware datetime
        if not isinstance(funding.index, pd.DatetimeIndex):
            funding.index = pd.to_datetime(funding.index, format="ISO8601", utc=True)
        elif funding.index.tz is None:
            funding.index = funding.index.tz_localize("UTC")
        log.info(f"  Loaded {len(funding)} funding records")
    
    fear_greed = None
    if fear_greed_path:
        log.info(f"Loading Fear & Greed from {fear_greed_path}")
        fg_df = pd.read_csv(fear_greed_path, index_col=0, parse_dates=True)
        fear_greed = fg_df["value"] if "value" in fg_df.columns else fg_df.iloc[:, 0]
        log.info(f"  Loaded {len(fear_greed)} Fear & Greed records")
    
    return df, funding, fear_greed


def create_labels(
    ohlcv: pd.DataFrame,
    threshold: float = 25.0,
    lookahead: int = 16,
) -> pd.Series:
    """
    Create labels for regime classification.
    
    Label = 1 if the FUTURE regime score > threshold (Trend)
    Label = 0 otherwise (Mean Reversion)
    
    Args:
        ohlcv: OHLCV DataFrame
        threshold: ADX threshold for regime switch
        lookahead: Number of bars to look ahead (16 = 4 hours at 15m)
    
    Returns:
        Series of labels (0 or 1)
    """
    log.info(f"Creating labels (threshold={threshold}, lookahead={lookahead} bars)")
    
    # Calculate current regime score using existing ADX logic
    regime_score = get_regime_score(ohlcv)
    
    # Look AHEAD to create labels (we want to predict future regime)
    future_score = regime_score.shift(-lookahead)
    
    # Binary label: 1 = Trend, 0 = Mean Reversion
    labels = (future_score > threshold).astype(int)
    
    # Log class distribution
    n_trend = labels.sum()
    n_mr = len(labels) - n_trend - labels.isna().sum()
    log.info(f"  Label distribution: {n_trend} Trend (1), {n_mr} MR (0)")
    
    return labels


def train_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple[list[float], list[dict]]:
    """
    Walk-forward cross-validation.
    
    This is critical for time series to detect overfitting.
    Each fold trains on past data and tests on future data.
    
    Returns:
        List of accuracy scores for each fold
        List of detailed metrics dicts for each fold
    """
    log.info(f"Running {n_splits}-fold walk-forward cross-validation")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    detailed_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Drop NaN labels
        train_mask = ~y_train.isna()
        test_mask = ~y_test.isna()
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        
        if len(X_test) == 0:
            log.warning(f"  Fold {fold+1}: No test samples, skipping")
            continue
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=50,  # Regularization
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        scores.append(acc)
        detailed_metrics.append({
            "fold": fold + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })
        
        log.info(f"  Fold {fold+1}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    log.info(f"  Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    
    return scores, detailed_metrics


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[RandomForestClassifier, StandardScaler]:
    """
    Train final model on all available data.
    
    Returns:
        Trained RandomForestClassifier
        Fitted StandardScaler
    """
    log.info("Training final model on full dataset")
    
    # Drop NaN
    mask = ~y.isna()
    X_clean = X[mask]
    y_clean = y[mask]
    
    log.info(f"  Training samples: {len(X_clean)}")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_clean)
    
    log.info("  Model trained successfully")
    
    return model, scaler


def get_feature_importance(model: RandomForestClassifier, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from trained model."""
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return importance


def save_training_report(
    output_dir: Path,
    cv_scores: list[float],
    cv_metrics: list[dict],
    feature_importance: pd.DataFrame,
    model_params: dict,
):
    """Save training report as Markdown."""
    report_path = output_dir / "training_report_v1.md"
    
    with open(report_path, "w") as f:
        f.write("# ML Regime Classifier - Training Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        
        f.write("## Model Parameters\n\n")
        for k, v in model_params.items():
            f.write(f"- **{k}**: {v}\n")
        
        f.write("\n## Walk-Forward Cross-Validation Results\n\n")
        f.write(f"**Mean Accuracy**: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})\n\n")
        
        f.write("| Fold | Train Size | Test Size | Accuracy | Precision | Recall | F1 |\n")
        f.write("|------|------------|-----------|----------|-----------|--------|----|\n")
        for m in cv_metrics:
            f.write(f"| {m['fold']} | {m['train_size']} | {m['test_size']} | "
                    f"{m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n")
        
        f.write("\n## Feature Importance\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for i, row in feature_importance.iterrows():
            f.write(f"| {i+1} | {row['feature']} | {row['importance']:.4f} |\n")
        
        f.write("\n## Interpretation Guide\n\n")
        f.write("> [!NOTE]\n")
        f.write("> - **Accuracy > 55%** is acceptable (random baseline is 50%)\n")
        f.write("> - **Decreasing accuracy** across folds may indicate regime drift\n")
        f.write("> - **Top features should include ADX** (it's the current baseline)\n")
    
    log.info(f"Saved training report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ML Regime Classifier")
    
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--funding", default=None, help="Path to funding rate CSV")
    parser.add_argument("--fear-greed", default=None, help="Path to Fear & Greed CSV")
    parser.add_argument("--output", default="models/regime_classifier_v1.pkl", 
                        help="Output path for trained model")
    parser.add_argument("--threshold", type=float, default=25.0,
                        help="ADX threshold for regime classification")
    parser.add_argument("--lookahead", type=int, default=16,
                        help="Bars to look ahead for labels (16 = 4h at 15m)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds")
    
    args = parser.parse_args()
    
    # Load data
    df, funding, fear_greed = load_data(
        args.data, args.funding, args.fear_greed
    )
    
    # Build features
    log.info("Building features...")
    features = build_regime_features(df, funding, fear_greed)
    log.info(f"  Features shape: {features.shape}")
    
    # Create labels
    labels = create_labels(df, threshold=args.threshold, lookahead=args.lookahead)
    
    # Align features and labels
    common_idx = features.index.intersection(labels.dropna().index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    log.info(f"  Aligned samples: {len(X)}")
    
    # Walk-forward CV
    cv_scores, cv_metrics = train_walk_forward(X, y, n_splits=args.cv_folds)
    
    # Check if model is worth saving
    mean_acc = np.mean(cv_scores)
    if mean_acc < 0.52:
        log.warning(f"Mean accuracy ({mean_acc:.4f}) is barely above random. Consider revising features.")
    
    # Train final model
    model, scaler = train_final_model(X, y)
    
    # Feature importance
    feature_importance = get_feature_importance(model, get_feature_names())
    log.info("Feature Importance:")
    for _, row in feature_importance.head(5).iterrows():
        log.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model and scaler
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, output_path)
    log.info(f"Saved model to {output_path}")
    
    scaler_path = output_path.parent / output_path.name.replace("classifier", "scaler")
    joblib.dump(scaler, scaler_path)
    log.info(f"Saved scaler to {scaler_path}")
    
    # Save training report
    model_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 50,
        "threshold": args.threshold,
        "lookahead_bars": args.lookahead,
        "training_samples": len(X) - labels.isna().sum(),
        "features": get_feature_names(),
    }
    save_training_report(output_path.parent, cv_scores, cv_metrics, feature_importance, model_params)
    
    print(f"\n✅ Training complete!")
    print(f"   Model: {output_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Report: {output_path.parent / 'training_report_v1.md'}")
    print(f"   Mean CV Accuracy: {mean_acc:.4f}")


if __name__ == "__main__":
    main()

# ML Regime Classifier - Training Report

**Generated**: 2025-12-08T23:17:58.211518

## Model Parameters

- **n_estimators**: 100
- **max_depth**: 10
- **min_samples_leaf**: 50
- **threshold**: 25.0
- **lookahead_bars**: 16
- **training_samples**: 166393
- **features**: ['adx_15m', 'rsi_14', 'volume_ratio', 'bb_width', 'roc_4h', 'returns_std', 'funding_rate', 'funding_zscore', 'fear_greed']

## Walk-Forward Cross-Validation Results

**Mean Accuracy**: 0.8398 (+/- 0.0228)

| Fold | Train Size | Test Size | Accuracy | Precision | Recall | F1 |
|------|------------|-----------|----------|-----------|--------|----|
| 1 | 27733 | 27732 | 0.8690 | 0.7402 | 0.0696 | 0.1273 |
| 2 | 55465 | 27732 | 0.8511 | 0.5132 | 0.1313 | 0.2091 |
| 3 | 83197 | 27732 | 0.8529 | 0.6842 | 0.0764 | 0.1374 |
| 4 | 110929 | 27732 | 0.8116 | 0.7440 | 0.1109 | 0.1930 |
| 5 | 138661 | 27732 | 0.8143 | 0.6643 | 0.1351 | 0.2246 |

## Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | adx_15m | 0.5981 |
| 7 | funding_rate | 0.0790 |
| 9 | fear_greed | 0.0730 |
| 6 | returns_std | 0.0695 |
| 2 | rsi_14 | 0.0666 |
| 8 | funding_zscore | 0.0456 |
| 5 | roc_4h | 0.0277 |
| 4 | bb_width | 0.0276 |
| 3 | volume_ratio | 0.0129 |

## Interpretation Guide

> [!NOTE]
> - **Accuracy > 55%** is acceptable (random baseline is 50%)
> - **Decreasing accuracy** across folds may indicate regime drift
> - **Top features should include ADX** (it's the current baseline)

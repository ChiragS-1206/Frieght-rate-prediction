# Freight Rate Prediction

## Overview
This project predicts freight rates using machine learning.

## Approach
- EDA on numeric and categorical features
- Strong correlation found with distance
- Used RandomForestRegressor with preprocessing pipeline

## Model
- Algorithm: Random Forest
- Features: distance, location, equipment, market_index, etc.
- Evaluation:
  - MAE: 38.32
  - R²: 0.997

## Files
- validation_predictions.csv → final predictions
- december_chart_inputs.csv → December predictions
- scorer_results/candidate_december.png → output chart

## How to run

```bash
pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
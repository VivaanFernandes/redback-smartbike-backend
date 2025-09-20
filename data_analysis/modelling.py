"""
modelling.py
Handles regression modelling for workout analysis.
Now integrates Lachesis API as primary analysis,
with sklearn LinearRegression as a fallback.
"""

from sklearn.linear_model import LinearRegression
from .lachesis_api import analyze


def train_regression_model(X, y):
    """
    Train a regression model for workout analysis.

    Process:
    1. Try sending the dataset to Lachesis API for advanced analysis.
    2. If Lachesis is unavailable or fails, fall back to a local sklearn LinearRegression.

    Args:
        X (array-like): Features (e.g. speed values).
        y (array-like): Target values (e.g. heart rates).

    Returns:
        dict | LinearRegression: 
            - Lachesis result dict if API call succeeds.
            - sklearn LinearRegression model if fallback is used.
    """
    payload = {
        "metric": "heart_rate_prediction",
        "features": X,
        "target": y
    }

    # Attempt Lachesis API
    result = analyze(payload)
    if result:
        print("[modelling] Using Lachesis API results")
        return result

    # Fallback to sklearn
    print("[modelling] Lachesis unavailable, falling back to local LinearRegression")
    model = LinearRegression().fit(X, y)
    return model

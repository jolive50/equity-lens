"""Test script to verify ensemble evaluation works with evaluate_model.py.

Creates a mock ensemble with simple predictors and tests the evaluation pipeline.
"""
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.prediction.ensemble import PredictionEnsemble
from models.prediction.mock_predictor import MockPredictor


def create_test_data(n_samples=1000, n_features=5):
    """Create synthetic test data.

    Args:
        n_samples: Number of samples
        n_features: Number of features (OHLCV = 5)

    Returns:
        Tuple of (X_test, y_test)
    """
    print(f"Creating test data: {n_samples} samples, {n_features} features")

    # Generate random OHLCV-like data
    X_test = np.random.randn(n_samples, n_features)

    # Generate labels with class imbalance (typical in financial data)
    # 40% neutral, 30% up, 30% down
    y_test = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])

    return X_test, y_test


def main():
    """Main test function."""
    print("=" * 60)
    print("Testing Ensemble Evaluation Pipeline")
    print("=" * 60)

    # Create mock predictors
    print("\n1. Creating mock predictors...")
    predictor1 = MockPredictor("MockPredictor1", bias="up")
    predictor2 = MockPredictor("MockPredictor2", bias="down")
    predictor3 = MockPredictor("MockPredictor3", bias="neutral")

    # Create ensemble
    print("2. Creating ensemble with weighted_average strategy...")
    ensemble = PredictionEnsemble(
        models=[predictor1, predictor2, predictor3],
        strategy="weighted_average",
        weights={
            "MockPredictor1": 0.4,
            "MockPredictor2": 0.3,
            "MockPredictor3": 0.3
        }
    )

    # Save ensemble
    ensemble_path = Path("models/prediction/saved_models/ensemble")
    ensemble_path.mkdir(parents=True, exist_ok=True)
    ensemble_file = ensemble_path / "test_ensemble.pkl"

    print(f"3. Saving ensemble to {ensemble_file}...")
    with open(ensemble_file, 'wb') as f:
        pickle.dump(ensemble, f)

    # Create test data
    print("\n4. Creating test data...")
    X_test, y_test = create_test_data(n_samples=500, n_features=5)

    # Save test data in the expected format
    data_path = Path("data/processed")
    data_path.mkdir(parents=True, exist_ok=True)
    test_data_file = data_path / "test_splits_ensemble.npz"

    print(f"5. Saving test data to {test_data_file}...")
    np.savez(
        test_data_file,
        X_lstm_test=X_test,
        y_lstm_test=y_test,
        X_xgb_test=X_test,
        y_xgb_test=y_test
    )

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nEnsemble saved to: {ensemble_file}")
    print(f"Test data saved to: {test_data_file}")
    print("\nTo evaluate the ensemble, run:")
    print(f"python -m models.prediction.evaluate_model \\")
    print(f"    --model-path {ensemble_file} \\")
    print(f"    --model-type ensemble \\")
    print(f"    --data-path {test_data_file}")
    print("\nOr with visualizations:")
    print(f"python -m models.prediction.evaluate_model \\")
    print(f"    --model-path {ensemble_file} \\")
    print(f"    --model-type ensemble \\")
    print(f"    --data-path {test_data_file} \\")
    print(f"    --visualize")

    return 0


if __name__ == "__main__":
    sys.exit(main())

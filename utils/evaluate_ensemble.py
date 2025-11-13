"""Script to create and evaluate an ensemble from trained models.

This script:
1. Loads your trained models (LSTM, GRU, XGBoost, etc.)
2. Wraps them in BasePredictionModel wrappers
3. Creates a PredictionEnsemble
4. Evaluates the ensemble using the evaluate_model.py functions

Usage:
    python evaluate_ensemble.py
    python evaluate_ensemble.py --strategy voting
    python evaluate_ensemble.py --visualize
"""
import argparse
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Import prediction components
from models.prediction.base_predictor import BasePredictionModel, PredictionResult
from models.prediction.ensemble import PredictionEnsemble
from utils.evaluate_model import (
    load_test_data,
    evaluate_model,
    visualize_results
)


class KerasModelWrapper(BasePredictionModel):
    """Wrapper for Keras models (LSTM, GRU) to work with ensemble."""

    def __init__(self, model_path: str, name: str):
        """Initialize wrapper.

        Args:
            model_path: Path to saved Keras model
            name: Name for this model
        """
        import tensorflow as tf
        from tensorflow import keras

        self.model = keras.models.load_model(model_path)
        self.name = name
        self.model_path = model_path

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make prediction from DataFrame input.

        Args:
            data: DataFrame with OHLCV data (single row or multiple rows)

        Returns:
            PredictionResult with direction and probabilities
        """
        # Convert DataFrame to numpy array
        # For sequential models, we need to reshape to 3D
        X = data.values

        # If we have multiple rows, use them as sequence
        # Otherwise, create a sequence of 1 timestep
        if len(X.shape) == 1:
            X = X.reshape(1, 1, -1)  # (1, 1, features)
        elif len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])  # (1, timesteps, features)

        # Get prediction probabilities
        probs = self.model.predict(X, verbose=0)[0]

        # Get direction
        class_idx = np.argmax(probs)
        directions = ['down', 'neutral', 'up']
        direction = directions[class_idx]

        return PredictionResult(
            direction=direction,
            confidence=float(probs[class_idx]),
            probabilities={
                'down': float(probs[0]),
                'neutral': float(probs[1]),
                'up': float(probs[2])
            },
            metadata={'model': self.name, 'type': 'keras'}
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            'name': self.name,
            'type': 'keras',
            'path': self.model_path
        }


class SklearnModelWrapper(BasePredictionModel):
    """Wrapper for sklearn-style models (XGBoost, GradientBoosting) to work with ensemble."""

    def __init__(self, model_path: str, name: str):
        """Initialize wrapper.

        Args:
            model_path: Path to saved pickle model
            name: Name for this model
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.name = name
        self.model_path = model_path

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make prediction from DataFrame input.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            PredictionResult with direction and probabilities
        """
        # Convert DataFrame to numpy array
        X = data.values

        # Flatten if needed (sklearn models expect 2D)
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        elif len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Get prediction
        pred_class = self.model.predict(X)[0]

        # Try to get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[0]
        else:
            # Create one-hot probabilities
            probs = np.zeros(3)
            probs[pred_class] = 1.0

        # Get direction
        directions = ['down', 'neutral', 'up']
        direction = directions[pred_class]

        return PredictionResult(
            direction=direction,
            confidence=float(probs[pred_class]),
            probabilities={
                'down': float(probs[0]),
                'neutral': float(probs[1]),
                'up': float(probs[2])
            },
            metadata={'model': self.name, 'type': 'sklearn'}
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            'name': self.name,
            'type': 'sklearn',
            'path': self.model_path
        }


def load_ensemble_models(model_paths: Dict[str, str]) -> list:
    """Load and wrap models for ensemble.

    Args:
        model_paths: Dictionary mapping model names to file paths

    Returns:
        List of wrapped BasePredictionModel instances
    """
    models = []

    for name, path in model_paths.items():
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"Warning: Model not found at {path}, skipping...")
            continue

        print(f"Loading {name} from {path}...")

        if path.endswith('.keras') or path.endswith('.h5') or path.endswith('.hdf5'):
            # Keras model
            wrapper = KerasModelWrapper(path, name)
            models.append(wrapper)
        elif path.endswith('.pkl') or path.endswith('.pickle'):
            # Sklearn-style model
            wrapper = SklearnModelWrapper(path, name)
            models.append(wrapper)
        else:
            print(f"Warning: Unknown file format for {path}, skipping...")

    return models


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Create and evaluate ensemble from trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='weighted_average',
        choices=['weighted_average', 'simple_average', 'voting'],
        help='Ensemble strategy (default: weighted_average)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='Custom weights as comma-separated values (e.g., "0.4,0.3,0.3")'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/training_splits.npz',
        help='Path to test data file'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--save-ensemble',
        action='store_true',
        help='Save the ensemble to a pickle file'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Ensemble Model Evaluation")
    print("=" * 60)

    # Define your trained models
    # NOTE: For the ensemble to work, models must accept the same input format
    # LSTM/GRU expect 3D sequential data (samples, timesteps, features)
    # XGBoost/GradientBoost expect 2D tabular data (samples, features)
    #
    # The current models were trained on different feature sets:
    # - LSTM/GRU: 25 features per timestep
    # - XGBoost: 28 features
    # - GradientBoost: 10 features
    #
    # For now, we'll only use XGBoost since it matches the 28-feature test data
    # To use all models together, you'd need to retrain them on the same features
    # or create feature-specific wrappers

    model_paths = {
        'XGBoost': 'models/prediction/saved_models/xgboost/best_model.pkl',
    }

    # Uncomment to add more models once feature alignment is resolved
    # model_paths['LSTM'] = 'models/prediction/saved_models/lstm/best_model.keras'
    # model_paths['GRU'] = 'models/prediction/saved_models/gru/best_model.keras'
    # model_paths['GradientBoost'] = 'models/prediction/saved_models/gradient_boost/gb_model.pkl'

    print("\nStep 1: Loading models...")
    models = load_ensemble_models(model_paths)

    if len(models) < 2:
        print(f"\nError: Need at least 2 models for ensemble, only found {len(models)}")
        print("Available models:")
        for name, path in model_paths.items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {exists} {name}: {path}")
        return 1

    print(f"\nLoaded {len(models)} models:")
    for model in models:
        info = model.get_model_info()
        print(f"  - {info['name']} ({info['type']})")

    # Parse weights if provided
    weights = None
    if args.weights:
        weight_values = [float(w) for w in args.weights.split(',')]
        if len(weight_values) != len(models):
            print(f"\nError: Number of weights ({len(weight_values)}) doesn't match number of models ({len(models)})")
            return 1
        weights = {models[i].get_model_info()['name']: weight_values[i] for i in range(len(models))}
        print(f"\nUsing custom weights: {weights}")

    # Create ensemble
    print(f"\nStep 2: Creating ensemble with strategy '{args.strategy}'...")
    ensemble = PredictionEnsemble(
        models=models,
        strategy=args.strategy,
        weights=weights
    )

    # Save ensemble if requested
    if args.save_ensemble:
        ensemble_dir = Path('models/prediction/saved_models/ensemble')
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        ensemble_path = ensemble_dir / f'ensemble_{args.strategy}.pkl'

        print(f"\nSaving ensemble to {ensemble_path}...")
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble, f)
        print(f"  Ensemble saved!")

    # Load test data
    print(f"\nStep 3: Loading test data from {args.data_path}...")
    try:
        # For ensemble, we'll use the XGBoost format (2D) as it's simpler
        test_data = load_test_data(args.data_path, 'xgboost')
        X_test = test_data['X_test']
        y_test = test_data['y_test']
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have preprocessed data. Run:")
        print("  python -m data.preprocess_data")
        return 1

    # Evaluate ensemble
    print("\nStep 4: Evaluating ensemble...")
    results = evaluate_model(
        model=ensemble,
        X_test=X_test,
        y_test=y_test,
        model_type='ensemble',
        batch_size=64,
        save_dir='models/prediction/saved_models/ensemble'
    )

    # Visualize if requested
    if args.visualize:
        print("\nStep 5: Creating visualizations...")
        try:
            visualize_results(results, save_dir='models/prediction/saved_models/ensemble')
        except ImportError as e:
            print(f"\nWarning: Could not create visualizations: {e}")
            print("Install matplotlib and seaborn to enable visualizations:")
            print("  pip install matplotlib seaborn")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

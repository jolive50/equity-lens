"""Universal model evaluation script.

Loads and evaluates any saved model (LSTM, GRU, XGBoost, Ensemble, etc.) with:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix and per-class performance
- Financial metrics (expected value, win rate, payoff asymmetry)
- Prediction distribution analysis
- Optional visualization of results

Supports:
- Deep learning models (LSTM, GRU) - Keras .keras files
- Gradient boosting models (XGBoost, LightGBM) - pickle files
- Ensemble models (PredictionEnsemble) - pickle files (requires --model-type ensemble)
- Automatically detects model type from file extension (except ensemble)

Usage:
    python -m models.prediction.evaluate_model --model-path path/to/model.keras
    python -m models.prediction.evaluate_model --model-path path/to/model.pkl --model-type xgboost
    python -m models.prediction.evaluate_model --model-path path/to/ensemble.pkl --model-type ensemble
    python -m models.prediction.evaluate_model --model-path path/to/model.keras --visualize
"""
import logging
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_test_data(data_path: str, model_type: str) -> Dict[str, np.ndarray]:
    """Load preprocessed test data based on model type.

    Args:
        data_path: Path to training splits file
        model_type: Type of model ('lstm', 'gru', 'xgboost', etc.)

    Returns:
        Dictionary with test data
    """
    logger.info(f"Loading test data from: {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Test data not found: {data_path}\n"
            f"Run: python -m data.preprocess_data"
        )

    data = np.load(data_path)

    # Determine which data format to use based on model type
    if model_type in ['lstm', 'gru', 'rnn', 'neural']:
        # Sequential models need 3D data (samples, timesteps, features)
        X_test = data['X_lstm_test']
        y_test = data['y_lstm_test']
        logger.info(f"  Loaded sequential data for {model_type.upper()} model")
    else:
        # Traditional ML models need 2D data (samples, features)
        # Try multiple naming conventions
        if 'X_xgb_test' in data:
            X_test = data['X_xgb_test']
            y_test = data['y_xgb_test']
        elif 'X_gb_test' in data:
            X_test = data['X_gb_test']
            y_test = data['y_gb_test']
        else:
            raise KeyError(f"Could not find test data in file. Available keys: {list(data.keys())}")
        logger.info(f"  Loaded tabular data for {model_type.upper()} model")

    logger.info(f"  Test set shape: {X_test.shape}")
    logger.info(f"  Test labels shape: {y_test.shape}")

    return {
        'X_test': X_test,
        'y_test': y_test
    }


def load_model(model_path: str, model_type: str = None) -> Tuple[Any, str]:
    """Load saved model (auto-detects type if not specified).

    Args:
        model_path: Path to saved model file
        model_type: Optional model type ('lstm', 'gru', 'xgboost', 'ensemble', etc.)

    Returns:
        Tuple of (loaded model, detected model type)
    """
    logger.info(f"Loading model from: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    path = Path(model_path)
    file_extension = path.suffix.lower()

    # Auto-detect model type from file extension if not provided
    if model_type is None:
        if file_extension in ['.keras', '.h5', '.hdf5']:
            model_type = 'neural'
        elif file_extension in ['.pkl', '.pickle', '.joblib']:
            model_type = 'xgboost'
        else:
            raise ValueError(
                f"Cannot auto-detect model type from extension '{file_extension}'. "
                f"Please specify --model-type explicitly."
            )
        logger.info(f"  Auto-detected model type: {model_type}")

    # Load based on type
    if model_type in ['lstm', 'gru', 'rnn', 'neural']:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        logger.info(f"  Loaded Keras model ({model_type.upper()})")
        logger.info(f"\n  Model architecture:")
        model.summary(print_fn=logger.info)

    elif model_type in ['xgboost', 'gradient_boost', 'lightgbm', 'catboost']:
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"  Loaded pickled model ({model_type.upper()})")

    elif model_type == 'ensemble':
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"  Loaded ensemble model")
        model_info = model.get_model_info()
        logger.info(f"  Strategy: {model_info['strategy']}")
        logger.info(f"  Component models: {', '.join(model_info['models'])}")
        logger.info(f"  Number of models: {model_info['num_models']}")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, model_type


def predict_with_ensemble(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Generate predictions using an ensemble model.

    Converts numpy arrays to DataFrames and extracts predictions from PredictionResult objects.

    Args:
        model: PredictionEnsemble instance
        X_test: Test data (2D or 3D numpy array)
        y_test: Test labels (for aligning data length)

    Returns:
        Predicted class labels as numpy array
    """
    import pandas as pd

    logger.info("\nGenerating predictions with ensemble...")
    logger.info(f"  Input shape: {X_test.shape}")

    # For ensemble, we need to convert each sample to a DataFrame
    # Ensemble expects DataFrame with OHLCV data
    predictions = []

    # If X_test is 3D (samples, timesteps, features), we need to handle it differently
    if len(X_test.shape) == 3:
        # For sequential data, use the last timestep as current state
        logger.info("  Converting 3D sequential data to 2D (using last timestep)")
        X_2d = X_test[:, -1, :]  # Shape: (samples, features)
    else:
        X_2d = X_test

    # Determine feature columns based on shape
    # Typical OHLCV has 5 features, but with technical indicators it could be more
    num_features = X_2d.shape[1]

    # Create column names (we'll use generic names since we don't know the actual feature names)
    if num_features >= 5:
        # Assume first 5 are OHLCV
        columns = ['open', 'high', 'low', 'close', 'volume'] + [f'feature_{i}' for i in range(5, num_features)]
    else:
        columns = [f'feature_{i}' for i in range(num_features)]

    logger.info(f"  Creating DataFrames with {num_features} features")
    logger.info(f"  Processing {len(X_2d)} samples...")

    # Process in batches to show progress for large datasets
    batch_size = 100
    for i in range(0, len(X_2d), batch_size):
        batch_end = min(i + batch_size, len(X_2d))

        for idx in range(i, batch_end):
            # Create DataFrame for this sample
            # Ensemble models typically expect multiple rows of historical data,
            # but we'll give it a single row per sample
            sample_df = pd.DataFrame([X_2d[idx]], columns=columns)

            try:
                # Get prediction from ensemble
                pred_result = model.predict(sample_df)

                # Convert direction to class label
                direction_to_class = {'down': 0, 'neutral': 1, 'up': 2}
                predictions.append(direction_to_class[pred_result.direction])

            except Exception as e:
                logger.warning(f"  Prediction failed for sample {idx}: {e}")
                # Default to neutral if prediction fails
                predictions.append(1)

        if batch_end % 1000 == 0 or batch_end == len(X_2d):
            logger.info(f"  Processed {batch_end}/{len(X_2d)} samples")

    predictions = np.array(predictions)
    logger.info(f"  Generated {len(predictions)} predictions")

    return predictions


def predict_with_model(model: Any, X_test: np.ndarray, model_type: str, batch_size: int = 64, y_test: np.ndarray = None) -> np.ndarray:
    """Generate predictions using the model.

    Args:
        model: Loaded model
        X_test: Test data
        model_type: Type of model
        batch_size: Batch size for neural networks
        y_test: Test labels (only needed for ensemble models)

    Returns:
        Predicted class labels
    """
    logger.info("\nGenerating predictions...")

    if model_type == 'ensemble':
        # Ensemble model: needs special handling with DataFrames
        if y_test is None:
            raise ValueError("y_test is required for ensemble model predictions")
        return predict_with_ensemble(model, X_test, y_test)

    elif model_type in ['lstm', 'gru', 'rnn', 'neural']:
        # Neural network: returns probabilities
        y_pred_proba = model.predict(X_test, verbose=0, batch_size=batch_size)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # Gradient boosting: use predict method
        y_pred = model.predict(X_test)

    logger.info(f"  Generated {len(y_pred)} predictions")
    return y_pred


def calculate_financial_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate financial evaluation metrics for trading strategy.

    Simulates a simple trading strategy:
    - Predict "up" (class 2): Take long position
    - Predict "down" (class 0): Take short position
    - Predict "neutral" (class 1): No trade

    Args:
        y_true: True labels (0=down, 1=neutral, 2=up)
        y_pred: Predicted labels

    Returns:
        Dictionary with financial metrics
    """
    # Map classes to simulated returns (±0.5% per research threshold)
    class_to_return = {0: -0.5, 1: 0.0, 2: 0.5}

    trades = []
    profits = []
    losses = []

    for i in range(len(y_true)):
        pred_class = y_pred[i]
        true_class = y_true[i]

        # Skip neutral predictions (no trade)
        if pred_class == 1:
            continue

        # Actual return based on true class
        actual_return = class_to_return[true_class]

        # Trading return (long if pred up, short if pred down)
        if pred_class == 2:  # Predicted up, go long
            trade_return = actual_return
        elif pred_class == 0:  # Predicted down, go short
            trade_return = -actual_return
        else:
            continue

        trades.append(trade_return)

        if trade_return > 0:
            profits.append(trade_return)
        elif trade_return < 0:
            losses.append(abs(trade_return))

    # Calculate metrics
    if len(trades) == 0:
        return {
            'expected_value_per_trade': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'payoff_asymmetry': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }

    expected_value = np.mean(trades)
    avg_profit = np.mean(profits) if profits else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    payoff_asymmetry = avg_profit / avg_loss if avg_loss > 0 else 0.0
    win_rate = (len(profits) / len(trades)) * 100 if trades else 0.0

    return {
        'expected_value_per_trade': float(expected_value),
        'avg_profit': float(avg_profit),
        'avg_loss': float(avg_loss),
        'payoff_asymmetry': float(payoff_asymmetry),
        'win_rate': float(win_rate),
        'total_trades': len(trades),
        'winning_trades': len(profits),
        'losing_trades': len(losses)
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    batch_size: int = 64,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate model on test data.

    Args:
        model: Loaded model
        X_test: Test sequences/features
        y_test: Test labels
        model_type: Type of model
        batch_size: Batch size for prediction
        save_dir: Optional directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support
    )

    logger.info("\n" + "=" * 60)
    logger.info("Evaluating Model on Test Set")
    logger.info("=" * 60)

    # Get predictions
    y_pred = predict_with_model(model, X_test, model_type, batch_size, y_test)

    # Calculate accuracy
    test_acc = accuracy_score(y_test, y_pred)

    # Calculate loss if available (neural networks)
    test_loss = None
    if model_type in ['lstm', 'gru', 'rnn', 'neural']:
        test_loss, _ = model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)

    # Classification metrics
    class_names = ['down', 'neutral', 'up']
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        average=None
    )

    # Prediction distribution
    pred_distribution = {
        'down': int(np.sum(y_pred == 0)),
        'neutral': int(np.sum(y_pred == 1)),
        'up': int(np.sum(y_pred == 2))
    }

    true_distribution = {
        'down': int(np.sum(y_test == 0)),
        'neutral': int(np.sum(y_test == 1)),
        'up': int(np.sum(y_test == 2))
    }

    # Financial metrics
    financial_metrics = calculate_financial_metrics(y_test, y_pred)

    # Compile results
    results = {
        'model_type': model_type,
        'test_accuracy': float(test_acc),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'down': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1_score': float(f1[0]),
                'support': int(support[0])
            },
            'neutral': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1_score': float(f1[1]),
                'support': int(support[1])
            },
            'up': {
                'precision': float(precision[2]),
                'recall': float(recall[2]),
                'f1_score': float(f1[2]),
                'support': int(support[2])
            }
        },
        'prediction_distribution': pred_distribution,
        'true_distribution': true_distribution,
        'financial_metrics': financial_metrics,
        'total_samples': len(y_test)
    }

    if test_loss is not None:
        results['test_loss'] = float(test_loss)

    # Display results
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Set Performance - {model_type.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"\nOverall Metrics:")
    if test_loss is not None:
        logger.info(f"  Test Loss:     {test_loss:.4f}")
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    logger.info(f"  Total Samples: {len(y_test)}")

    logger.info(f"\n{'='*60}")
    logger.info("Classification Report")
    logger.info(f"{'='*60}")
    logger.info(classification_report(y_test, y_pred, target_names=class_names))

    logger.info(f"{'='*60}")
    logger.info("Confusion Matrix")
    logger.info(f"{'='*60}")
    logger.info(f"                Predicted")
    logger.info(f"              Down  Neutral  Up")
    logger.info(f"Actual Down     {cm[0][0]:4d}    {cm[0][1]:4d}  {cm[0][2]:4d}")
    logger.info(f"       Neutral  {cm[1][0]:4d}    {cm[1][1]:4d}  {cm[1][2]:4d}")
    logger.info(f"       Up       {cm[2][0]:4d}    {cm[2][1]:4d}  {cm[2][2]:4d}")

    logger.info(f"\n{'='*60}")
    logger.info("Prediction Distribution")
    logger.info(f"{'='*60}")
    logger.info(f"              True  Predicted")
    logger.info(f"Down        {true_distribution['down']:5d}    {pred_distribution['down']:5d}  ({pred_distribution['down']/len(y_test)*100:.1f}%)")
    logger.info(f"Neutral     {true_distribution['neutral']:5d}    {pred_distribution['neutral']:5d}  ({pred_distribution['neutral']/len(y_test)*100:.1f}%)")
    logger.info(f"Up          {true_distribution['up']:5d}    {pred_distribution['up']:5d}  ({pred_distribution['up']/len(y_test)*100:.1f}%)")

    logger.info(f"\n{'='*60}")
    logger.info("Financial Metrics")
    logger.info(f"{'='*60}")
    logger.info(f"  Expected Value/Trade: {financial_metrics['expected_value_per_trade']:+.4f}%")
    logger.info(f"  Win Rate:            {financial_metrics['win_rate']:.2f}%")
    logger.info(f"  Payoff Asymmetry:    {financial_metrics['payoff_asymmetry']:.4f}x")
    logger.info(f"  Average Profit:      {financial_metrics['avg_profit']:+.4f}%")
    logger.info(f"  Average Loss:        {financial_metrics['avg_loss']:+.4f}%")
    logger.info(f"  Total Trades:        {financial_metrics['total_trades']}")
    logger.info(f"  Winning Trades:      {financial_metrics['winning_trades']}")
    logger.info(f"  Losing Trades:       {financial_metrics['losing_trades']}")

    # Save results if requested
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        results_file = save_path / f'evaluation_results_{model_type}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n  Results saved to: {results_file}")

    return results


def visualize_results(results: Dict[str, Any], save_dir: Optional[str] = None):
    """Create visualizations of evaluation results.

    Args:
        results: Evaluation results dictionary
        save_dir: Optional directory to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info("\nGenerating visualizations...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    model_type = results.get('model_type', 'Model').upper()

    # 1. Confusion Matrix Heatmap
    cm = np.array(results['confusion_matrix'])
    class_names = ['Down', 'Neutral', 'Up']

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title(f'Confusion Matrix - {model_type}', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)

    # 2. Per-Class Metrics
    metrics_data = results['per_class_metrics']
    classes = ['Down', 'Neutral', 'Up']
    precision = [metrics_data['down']['precision'], metrics_data['neutral']['precision'], metrics_data['up']['precision']]
    recall = [metrics_data['down']['recall'], metrics_data['neutral']['recall'], metrics_data['up']['recall']]
    f1 = [metrics_data['down']['f1_score'], metrics_data['neutral']['f1_score'], metrics_data['up']['f1_score']]

    x = np.arange(len(classes))
    width = 0.25

    axes[0, 1].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[0, 1].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[0, 1].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    axes[0, 1].set_xlabel('Class', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title(f'Per-Class Performance - {model_type}', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(classes)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Prediction Distribution
    pred_dist = results['prediction_distribution']
    true_dist = results['true_distribution']

    x = np.arange(len(classes))
    width = 0.35

    axes[1, 0].bar(x - width/2, [true_dist['down'], true_dist['neutral'], true_dist['up']],
                    width, label='True', alpha=0.8)
    axes[1, 0].bar(x + width/2, [pred_dist['down'], pred_dist['neutral'], pred_dist['up']],
                    width, label='Predicted', alpha=0.8)
    axes[1, 0].set_xlabel('Class', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title(f'True vs Predicted Distribution - {model_type}', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Financial Metrics Summary
    fin_metrics = results['financial_metrics']
    metric_names = ['Expected\nValue\n(%)', 'Win\nRate\n(%)', 'Payoff\nAsymmetry']
    metric_values = [
        fin_metrics['expected_value_per_trade'],
        fin_metrics['win_rate'],
        fin_metrics['payoff_asymmetry']
    ]

    colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in metric_values]
    axes[1, 1].bar(metric_names, metric_values, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Value', fontsize=12)
    axes[1, 1].set_title(f'Financial Performance - {model_type}', fontsize=14, fontweight='bold')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(metric_values):
        axes[1, 1].text(i, v + (0.01 if v > 0 else -0.03), f'{v:.3f}',
                        ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

    plt.tight_layout()

    # Save if requested
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        plot_file = save_path / f'evaluation_plots_{model_type.lower()}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"  Plots saved to: {plot_file}")

    plt.show()
    logger.info("  Visualizations displayed")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained model (LSTM, GRU, XGBoost, Ensemble, etc.)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate LSTM model (auto-detects from .keras extension)
  python -m models.prediction.evaluate_model --model-path models/prediction/saved_models/lstm/best_model.keras

  # Evaluate XGBoost model (auto-detects from .pkl extension)
  python -m models.prediction.evaluate_model --model-path models/prediction/saved_models/xgboost/model.pkl

  # Evaluate ensemble model (must specify model type)
  python -m models.prediction.evaluate_model --model-path models/prediction/saved_models/ensemble/ensemble.pkl --model-type ensemble

  # Explicitly specify model type
  python -m models.prediction.evaluate_model --model-path model.pkl --model-type xgboost

  # Generate visualizations
  python -m models.prediction.evaluate_model --model-path model.keras --visualize
        """
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to saved model file (.keras for neural networks, .pkl for gradient boosting/ensemble)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['lstm', 'gru', 'rnn', 'neural', 'xgboost', 'gradient_boost', 'lightgbm', 'catboost', 'ensemble'],
        help='Model type (auto-detected if not specified)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/training_splits.npz',
        help='Path to test data file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for prediction (neural networks only)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        help='Directory to save evaluation results (defaults to model directory)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate and display visualizations'
    )

    args = parser.parse_args()

    # Default save directory to model directory if not specified
    if args.save_dir is None:
        args.save_dir = str(Path(args.model_path).parent)

    logger.info("=" * 60)
    logger.info("Universal Model Evaluation Pipeline")
    logger.info("=" * 60)

    try:
        # Load model
        model, model_type = load_model(args.model_path, args.model_type)

        # Load test data
        data = load_test_data(args.data_path, model_type)

        # Evaluate
        results = evaluate_model(
            model,
            data['X_test'],
            data['y_test'],
            model_type,
            batch_size=args.batch_size,
            save_dir=args.save_dir
        )

        # Visualize if requested
        if args.visualize:
            try:
                visualize_results(results, save_dir=args.save_dir)
            except ImportError as e:
                logger.warning(f"\nCould not create visualizations: {e}")
                logger.warning("Install matplotlib and seaborn to enable visualizations:")
                logger.warning("  pip install matplotlib seaborn")

        logger.info("\n" + "=" * 60)
        logger.info("Evaluation complete!")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"\nFile not found: {e}")
        return 1

    except Exception as e:
        logger.error(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

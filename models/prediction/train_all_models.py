"""Master training script that trains all models sequentially with proper cleanup.

Trains all three models (LSTM, GRU, XGBoost) with:
- Memory cleanup between models
- Error handling
- Comprehensive reporting
- Best practices throughout

Usage:
    # Train all models
    python -m models.prediction.train_all_models

    # Train specific models
    python -m models.prediction.train_all_models --models lstm gru

    # Skip XGBoost grid search (faster)
    python -m models.prediction.train_all_models --no-gridsearch
"""
import logging
import sys
import argparse
import gc
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def cleanup_memory():
    """Force garbage collection and clear Keras session."""
    try:
        import keras.backend as K
        K.clear_session()
    except:
        pass

    gc.collect()
    logger.info("  Memory cleanup complete")


def train_lstm(no_gridsearch: bool = False) -> Dict[str, Any]:
    """Train LSTM model."""
    from models.prediction.train_lstm import load_training_data, train_lstm_model

    logger.info("\n" + "=" * 80)
    logger.info("STARTING LSTM TRAINING")
    logger.info("=" * 80)

    try:
        data = load_training_data()
        results = train_lstm_model(
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val']
        )

        logger.info("\n✓ LSTM training completed successfully")
        return results

    except Exception as e:
        logger.error(f"\n✗ LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        cleanup_memory()


def train_gru(no_gridsearch: bool = False) -> Dict[str, Any]:
    """Train GRU model."""
    from models.prediction.train_gru import load_training_data, train_gru_model

    logger.info("\n" + "=" * 80)
    logger.info("STARTING GRU TRAINING")
    logger.info("=" * 80)

    try:
        data = load_training_data()
        results = train_gru_model(
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val']
        )

        logger.info("\n✓ GRU training completed successfully")
        return results

    except Exception as e:
        logger.error(f"\n✗ GRU training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        cleanup_memory()


def train_xgboost(no_gridsearch: bool = False) -> Dict[str, Any]:
    """Train XGBoost model."""
    from models.prediction.train_xgboost import load_training_data, train_xgboost_model

    logger.info("\n" + "=" * 80)
    logger.info("STARTING XGBOOST TRAINING")
    logger.info("=" * 80)

    try:
        data = load_training_data()
        results = train_xgboost_model(
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val'],
            use_gridsearch=not no_gridsearch
        )

        logger.info("\n✓ XGBoost training completed successfully")
        return results

    except Exception as e:
        logger.error(f"\n✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        cleanup_memory()


def save_combined_report(results: Dict[str, Dict[str, Any]], output_file: str = "models/prediction/training_report.txt"):
    """Save comprehensive training report with all model metrics.

    Args:
        results: Dictionary with metrics for each model
        output_file: Path to save report
    """
    report_path = Path(output_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = [
        "=" * 80,
        "FreshStart - Combined Model Training Report",
        "=" * 80,
        ""
    ]

    # Individual model reports
    for model_name, model_results in results.items():
        if model_results is None:
            continue

        report.append("=" * 80)
        report.append(f"{model_name} Model Results")
        report.append("=" * 80)
        report.append("")

        # Training info
        report.append("Training Information:")
        report.append("-" * 40)
        report.append(f"  Training time: {model_results['training_time_hours']:.2f} hours")

        if 'epochs_completed' in model_results:
            report.append(f"  Epochs completed: {model_results['epochs_completed']}")

        # Hyperparameters
        report.append("\nHyperparameters:")
        report.append("-" * 40)
        for key, value in model_results['hyperparameters'].items():
            report.append(f"  {key}: {value}")

        # Accuracy metrics
        report.append("\nAccuracy Metrics:")
        report.append("-" * 40)
        report.append(f"  Train accuracy: {model_results['train_accuracy']:.4f}")
        report.append(f"  Val accuracy:   {model_results['val_accuracy']:.4f}")

        # Classification report
        if 'classification_report' in model_results:
            report.append("\nPer-Class Performance:")
            report.append("-" * 40)
            cr = model_results['classification_report']
            for class_name in ['down', 'neutral', 'up']:
                if class_name in cr:
                    report.append(f"\n  {class_name.upper()}:")
                    report.append(f"    Precision: {cr[class_name]['precision']:.4f}")
                    report.append(f"    Recall:    {cr[class_name]['recall']:.4f}")
                    report.append(f"    F1-score:  {cr[class_name]['f1-score']:.4f}")

        # Confusion matrix
        if 'confusion_matrix' in model_results:
            cm = model_results['confusion_matrix']
            report.append("\nConfusion Matrix:")
            report.append("-" * 40)
            report.append("                    Predicted")
            report.append("              Down    Neutral    Up")
            report.append(f"Actual Down    {cm[0][0]:4d}      {cm[0][1]:4d}    {cm[0][2]:4d}")
            report.append(f"       Neutral {cm[1][0]:4d}      {cm[1][1]:4d}    {cm[1][2]:4d}")
            report.append(f"       Up      {cm[2][0]:4d}      {cm[2][1]:4d}    {cm[2][2]:4d}")

        # Financial metrics
        if 'financial_metrics' in model_results:
            report.append("\nFinancial Evaluation (Simulated Trading):")
            report.append("-" * 40)
            fin = model_results['financial_metrics']
            report.append(f"  Expected Value/Trade:  {fin['expected_value_per_trade']:+.4f}%")
            report.append(f"  Average Profit:        {fin['avg_profit']:.4f}%")
            report.append(f"  Average Loss:          {fin['avg_loss']:.4f}%")
            report.append(f"  Payoff Asymmetry:      {fin['payoff_asymmetry']:.4f}x")
            report.append(f"  Win Rate:              {fin['win_rate']:.2f}%")
            report.append(f"  Total Trades:          {fin['total_trades']}")
            report.append(f"  Winning Trades:        {fin['winning_trades']}")
            report.append(f"  Losing Trades:         {fin['losing_trades']}")

            # Interpretation
            report.append("\n  Interpretation:")
            if fin['expected_value_per_trade'] > 0:
                report.append(f"    ✓ POSITIVE expectancy: Strategy is profitable on average")
            else:
                report.append(f"    ✗ NEGATIVE expectancy: Strategy loses money on average")

            if fin['payoff_asymmetry'] > 1.0:
                report.append(f"    ✓ Winners larger than losers (good risk/reward)")
            else:
                report.append(f"    ✗ Losers larger than winners (poor risk/reward)")

        report.append("\n")

    # Summary comparison
    report.append("=" * 80)
    report.append("Model Comparison Summary")
    report.append("=" * 80)

    report.append("\nValidation Accuracy:")
    for model_name, model_results in results.items():
        if model_results and 'val_accuracy' in model_results:
            report.append(f"  {model_name:15s}: {model_results['val_accuracy']:.4f}")

    report.append("\nExpected Value per Trade:")
    for model_name, model_results in results.items():
        if model_results and 'financial_metrics' in model_results:
            ev = model_results['financial_metrics']['expected_value_per_trade']
            report.append(f"  {model_name:15s}: {ev:+.4f}%")

    report.append("\nPayoff Asymmetry (Profit/Loss Ratio):")
    for model_name, model_results in results.items():
        if model_results and 'financial_metrics' in model_results:
            pa = model_results['financial_metrics']['payoff_asymmetry']
            report.append(f"  {model_name:15s}: {pa:.4f}x")

    report.append("\nWin Rate:")
    for model_name, model_results in results.items():
        if model_results and 'financial_metrics' in model_results:
            wr = model_results['financial_metrics']['win_rate']
            report.append(f"  {model_name:15s}: {wr:.2f}%")

    report.append("\n" + "=" * 80)
    report.append("Recommendation")
    report.append("=" * 80)

    # Find best model
    best_model = None
    best_ev = -float('inf')
    for model_name, model_results in results.items():
        if model_results and 'financial_metrics' in model_results:
            ev = model_results['financial_metrics']['expected_value_per_trade']
            if ev > best_ev:
                best_ev = ev
                best_model = model_name

    if best_model:
        report.append(f"\nBest model by Expected Value: {best_model}")
        report.append(f"Expected Value per Trade: {best_ev:+.4f}%")
    else:
        report.append("\nNo models completed successfully.")

    report.append("\n" + "=" * 80)
    report.append("End of Report")
    report.append("=" * 80)

    report_text = "\n".join(report)

    with open(report_path, 'w') as f:
        f.write(report_text)

    logger.info(f"\n✓ Combined training report saved to: {report_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train all prediction models')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['lstm', 'gru', 'xgboost'],
        default=['lstm', 'gru', 'xgboost'],
        help='Models to train (default: all)'
    )
    parser.add_argument(
        '--no-gridsearch',
        action='store_true',
        help='Skip XGBoost grid search (use default hyperparameters)'
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FreshStart - Master Model Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"\nModels to train: {', '.join(args.models)}")
    logger.info(f"XGBoost grid search: {'disabled' if args.no_gridsearch else 'enabled'}")

    results = {}

    # Train models sequentially
    if 'lstm' in args.models:
        results['LSTM'] = train_lstm(args.no_gridsearch)

    if 'gru' in args.models:
        results['GRU'] = train_gru(args.no_gridsearch)

    if 'xgboost' in args.models:
        results['XGBoost'] = train_xgboost(args.no_gridsearch)

    # Save combined report
    if any(v is not None for v in results.values()):
        save_combined_report(results)

        logger.info("\n" + "=" * 80)
        logger.info("✓ Training pipeline complete!")
        logger.info("=" * 80)
        logger.info("\nModels saved to:")
        for model_name in results.keys():
            if results[model_name]:
                logger.info(f"  - models/prediction/saved_models/{model_name.lower()}/")
        logger.info("\nNext steps:")
        logger.info("1. Review training report: models/prediction/training_report.txt")
        logger.info("2. Test models on test set (create test script)")
        logger.info("3. Use in workflow: python -m coordinator.workflow")
        logger.info("=" * 80)

        return 0
    else:
        logger.error("\n✗ No models trained successfully")
        return 1


if __name__ == "__main__":
    sys.exit(main())

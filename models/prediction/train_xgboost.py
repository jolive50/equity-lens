"""XGBoost model training script with best practices.

Trains XGBoost (Gradient Boosting) model on preprocessed SP500 data with:
- GridSearchCV for hyperparameter tuning
- Early stopping
- Model checkpointing
- Comprehensive evaluation metrics

Usage:
    python -m models.prediction.train_xgboost

    # Skip grid search (use defaults)
    python -m models.prediction.train_xgboost --no-gridsearch
"""
import logging
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(data_path: str = "data/processed/training_splits.npz") -> Dict[str, np.ndarray]:
    """Load preprocessed training data.

    Args:
        data_path: Path to training splits file

    Returns:
        Dictionary with train/val/test splits
    """
    logger.info(f"Loading data from: {data_path}")

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Run: python -m data.preprocess_data"
        )

    data = np.load(data_path)

    logger.info("✓ Loaded training data:")
    logger.info(f"  Train: {data['X_gb_train'].shape}")
    logger.info(f"  Val:   {data['X_gb_val'].shape}")
    logger.info(f"  Test:  {data['X_gb_test'].shape}")

    return {
        'X_train': data['X_gb_train'],
        'y_train': data['y_gb_train'],
        'X_val': data['X_gb_val'],
        'y_val': data['y_gb_val'],
        'X_test': data['X_gb_test'],
        'y_test': data['y_gb_test']
    }


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


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_gridsearch: bool = True,
    hyperparameters: Dict[str, Any] = None,
    save_dir: str = "models/prediction/saved_models/xgboost"
) -> Dict[str, Any]:
    """Train XGBoost model with best practices.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        use_gridsearch: Whether to perform grid search for hyperparameters
        hyperparameters: Optional hyperparameters dict (used if use_gridsearch=False)
        save_dir: Directory to save model and results

    Returns:
        Dictionary with training results and metrics
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import classification_report, confusion_matrix

    except ImportError:
        logger.error("XGBoost not installed!")
        logger.error("Install with: pip install xgboost scikit-learn")
        raise

    logger.info("\n" + "=" * 60)
    logger.info("Training XGBoost Model")
    logger.info("=" * 60)

    start_time = time.time()

    if use_gridsearch:
        logger.info("\nPerforming GridSearchCV for hyperparameter tuning...")
        logger.info("(This may take a while...)")

        # Grid search parameter space
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1]
        }

        logger.info(f"\nGrid search space: {sum([len(v) for v in param_grid.values()])} parameters")
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        logger.info(f"Total combinations: {total_combinations}")
        logger.info(f"Using 3-fold CV: {total_combinations * 3} total fits")

        # Base model
        base_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Faster training
        )

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        # Fit
        grid_search.fit(X_train, y_train)

        # Best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        logger.info(f"\n✓ Grid search complete!")
        logger.info(f"  Best CV score: {grid_search.best_score_:.4f}")
        logger.info(f"\n✓ Best hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"    {key}: {value}")

    else:
        logger.info("\nTraining with default/provided hyperparameters (skipping grid search)...")

        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0.1,
                'objective': 'multi:softmax',
                'num_class': 3,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            }
        else:
            # Ensure required parameters are set
            hyperparameters['objective'] = 'multi:softmax'
            hyperparameters['num_class'] = 3
            hyperparameters['random_state'] = 42
            hyperparameters['n_jobs'] = -1
            hyperparameters['tree_method'] = 'hist'

        logger.info("\nHyperparameters:")
        for key, value in hyperparameters.items():
            logger.info(f"  {key}: {value}")

        # Build and train model
        model = xgb.XGBClassifier(**hyperparameters)

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

        best_params = hyperparameters

    training_time = time.time() - start_time

    # Evaluate
    logger.info("\nEvaluating on all datasets...")
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    # Classification report
    class_names = ['down', 'neutral', 'up']
    class_report = classification_report(y_val, val_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_val, val_pred)

    # Financial metrics
    financial_metrics = calculate_financial_metrics(y_val, val_pred)

    # Feature importance
    feature_importance = model.feature_importances_.tolist()

    # Results
    results = {
        'hyperparameters': best_params,
        'training_time_hours': training_time / 3600,
        'used_gridsearch': use_gridsearch,
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'financial_metrics': financial_metrics,
        'feature_importance': feature_importance
    }

    if use_gridsearch:
        results['best_cv_score'] = float(grid_search.best_score_)
        results['cv_results_summary'] = {
            'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist()[:10],
            'std_test_scores': grid_search.cv_results_['std_test_score'].tolist()[:10]
        }

    # Save model and results
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = save_path / "best_model.json"
    model.save_model(model_file)
    logger.info(f"\n✓ Model saved to: {model_file}")

    # Also save as pickle for compatibility
    import pickle
    pickle_file = save_path / "best_model.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✓ Also saved as: {pickle_file}")

    # Save results
    results_file = save_path / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info(f"\n✓ Training complete:")
    logger.info(f"  Training time: {training_time/3600:.2f} hours")
    logger.info(f"  Train accuracy: {train_acc:.4f}")
    logger.info(f"  Val accuracy: {val_acc:.4f}")

    logger.info(f"\n✓ Classification Report:")
    logger.info(classification_report(y_val, val_pred, target_names=class_names))

    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                Predicted")
    logger.info(f"              Down  Neutral  Up")
    logger.info(f"Actual Down     {cm[0][0]:4d}    {cm[0][1]:4d}  {cm[0][2]:4d}")
    logger.info(f"       Neutral  {cm[1][0]:4d}    {cm[1][1]:4d}  {cm[1][2]:4d}")
    logger.info(f"       Up       {cm[2][0]:4d}    {cm[2][1]:4d}  {cm[2][2]:4d}")

    logger.info(f"\n✓ Financial Metrics:")
    logger.info(f"  Expected Value/Trade: {financial_metrics['expected_value_per_trade']:+.4f}%")
    logger.info(f"  Win Rate: {financial_metrics['win_rate']:.2f}%")
    logger.info(f"  Payoff Asymmetry: {financial_metrics['payoff_asymmetry']:.4f}x")
    logger.info(f"  Total Trades: {financial_metrics['total_trades']}")

    logger.info(f"\n✓ Results saved to: {results_file}")

    return results


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument(
        '--no-gridsearch',
        action='store_true',
        help='Skip grid search and use default hyperparameters'
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("XGBoost Model Training Pipeline")
    logger.info("=" * 60)

    try:
        # Load data
        data = load_training_data()

        # Train model
        results = train_xgboost_model(
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val'],
            use_gridsearch=not args.no_gridsearch
        )

        logger.info("\n" + "=" * 60)
        logger.info("✓ XGBoost training complete!")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"\n✗ Data not found: {e}")
        return 1

    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

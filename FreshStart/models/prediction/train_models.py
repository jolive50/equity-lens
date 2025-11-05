"""Training script for all prediction models.

PAM's Component - Model Training Pipeline
Trains LSTM, GRU, and Gradient Boost models on preprocessed SP500 data.

Usage:
    python -m models.prediction.train_models
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(data_dir: str = "data/processed") -> Dict[str, np.ndarray]:
    """Load preprocessed training data.

    Args:
        data_dir: Directory with processed data

    Returns:
        Dictionary with train/val/test splits
    """
    data_path = Path(data_dir) / "training_splits.npz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            "Run first: python -m data.preprocess_data"
        )

    logger.info(f"Loading data from: {data_path}")
    data = np.load(data_path)

    splits = {key: data[key] for key in data.files}

    logger.info(f"✓ Loaded training data:")
    logger.info(f"  LSTM/GRU Train: {splits['X_lstm_train'].shape}")
    logger.info(f"  LSTM/GRU Val:   {splits['X_lstm_val'].shape}")
    logger.info(f"  LSTM/GRU Test:  {splits['X_lstm_test'].shape}")
    logger.info(f"  GB Train: {splits['X_gb_train'].shape}")
    logger.info(f"  GB Val:   {splits['X_gb_val'].shape}")
    logger.info(f"  GB Test:  {splits['X_gb_test'].shape}")

    return splits


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_dir: str = "models/prediction/saved_models/lstm"
) -> Dict[str, Any]:
    """Train LSTM model.

    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        save_dir: Directory to save model

    Returns:
        Dictionary with training metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("Training LSTM Model")
    logger.info("=" * 60)

    try:
        import tensorflow as tf
        from tensorflow import keras

        # Model architecture
        sequence_length, n_features = X_train.shape[1], X_train.shape[2]
        n_classes = 3  # down, neutral, up

        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(n_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)

        # Train model
        logger.info("\nTraining...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=128,
            verbose=1
        )

        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

        logger.info(f"\n✓ Training complete:")
        logger.info(f"  Train accuracy: {train_acc:.4f}")
        logger.info(f"  Val accuracy:   {val_acc:.4f}")

        # Save model
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "lstm_model.keras"
        model.save(model_file)
        logger.info(f"  Saved to: {model_file}")

        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss)
        }

    except ImportError:
        logger.error("TensorFlow not installed!")
        logger.error("Install with: pip install tensorflow")
        raise


def train_gru_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_dir: str = "models/prediction/saved_models/gru"
) -> Dict[str, Any]:
    """Train GRU model.

    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        save_dir: Directory to save model

    Returns:
        Dictionary with training metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("Training GRU Model")
    logger.info("=" * 60)

    try:
        import tensorflow as tf
        from tensorflow import keras

        # Model architecture (simpler than LSTM)
        sequence_length, n_features = X_train.shape[1], X_train.shape[2]
        n_classes = 3

        model = keras.Sequential([
            keras.layers.GRU(64, return_sequences=True, input_shape=(sequence_length, n_features)),
            keras.layers.Dropout(0.2),
            keras.layers.GRU(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(n_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)

        # Train model
        logger.info("\nTraining...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=128,
            verbose=1
        )

        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

        logger.info(f"\n✓ Training complete:")
        logger.info(f"  Train accuracy: {train_acc:.4f}")
        logger.info(f"  Val accuracy:   {val_acc:.4f}")

        # Save model
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "gru_model.keras"
        model.save(model_file)
        logger.info(f"  Saved to: {model_file}")

        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss)
        }

    except ImportError:
        logger.error("TensorFlow not installed!")
        logger.error("Install with: pip install tensorflow")
        raise


def train_gradient_boost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_dir: str = "models/prediction/saved_models/gradient_boost"
) -> Dict[str, Any]:
    """Train Gradient Boosting model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        save_dir: Directory to save model

    Returns:
        Dictionary with training metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("Training Gradient Boost Model")
    logger.info("=" * 60)

    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score

        # Model parameters
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=3,
            random_state=42
        )

        logger.info("Training XGBoost classifier...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        logger.info(f"\n✓ Training complete:")
        logger.info(f"  Train accuracy: {train_acc:.4f}")
        logger.info(f"  Val accuracy:   {val_acc:.4f}")

        # Save model
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "gb_model.json"
        model.save_model(model_file)
        logger.info(f"  Saved to: {model_file}")

        # Also save as pickle for compatibility
        import pickle
        pickle_file = save_path / "gb_model.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(model, f)

        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc)
        }

    except ImportError:
        logger.error("XGBoost not installed!")
        logger.error("Install with: pip install xgboost scikit-learn")
        raise


def save_training_report(metrics: Dict[str, Dict[str, Any]], output_file: str = "models/prediction/training_report.txt"):
    """Save training report with all model metrics.

    Args:
        metrics: Dictionary with metrics for each model
        output_file: Path to save report
    """
    report_path = Path(output_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = ["=" * 60, "FreshStart - Model Training Report", "=" * 60, ""]

    for model_name, model_metrics in metrics.items():
        report.append(f"\n{model_name} Model:")
        report.append("-" * 40)
        for metric_name, value in model_metrics.items():
            report.append(f"  {metric_name}: {value:.4f}" if isinstance(value, float) else f"  {metric_name}: {value}")

    report.append("\n" + "=" * 60)

    report_text = "\n".join(report)

    with open(report_path, 'w') as f:
        f.write(report_text)

    logger.info(f"\n✓ Training report saved to: {report_path}")
    logger.info(report_text)


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("FreshStart - Model Training Pipeline")
    logger.info("=" * 60)

    try:
        # Load data
        data = load_training_data()

        # Train all models
        metrics = {}

        # LSTM
        try:
            lstm_metrics = train_lstm_model(
                data['X_lstm_train'],
                data['y_lstm_train'],
                data['X_lstm_val'],
                data['y_lstm_val']
            )
            metrics['LSTM'] = lstm_metrics
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")

        # GRU
        try:
            gru_metrics = train_gru_model(
                data['X_lstm_train'],  # Same data as LSTM
                data['y_lstm_train'],
                data['X_lstm_val'],
                data['y_lstm_val']
            )
            metrics['GRU'] = gru_metrics
        except Exception as e:
            logger.error(f"GRU training failed: {e}")

        # Gradient Boost
        try:
            gb_metrics = train_gradient_boost_model(
                data['X_gb_train'],
                data['y_gb_train'],
                data['X_gb_val'],
                data['y_gb_val']
            )
            metrics['GradientBoost'] = gb_metrics
        except Exception as e:
            logger.error(f"Gradient Boost training failed: {e}")

        # Save report
        if metrics:
            save_training_report(metrics)

            logger.info("\n" + "=" * 60)
            logger.info("✓ All models trained successfully!")
            logger.info("=" * 60)
            logger.info("\nModel files saved to:")
            logger.info("  - models/prediction/saved_models/lstm/")
            logger.info("  - models/prediction/saved_models/gru/")
            logger.info("  - models/prediction/saved_models/gradient_boost/")
            logger.info("\nNext steps:")
            logger.info("1. Test models: python -m models.prediction.test_models")
            logger.info("2. Use in workflow: python -m coordinator.workflow")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("\n✗ No models trained successfully")
            return 1

    except FileNotFoundError as e:
        logger.error(f"\n✗ {e}")
        return 1

    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

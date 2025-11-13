"""GRU model training script with best practices.

Trains GRU model on preprocessed SP500 data with:
- Hyperparameter tuning
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Comprehensive evaluation metrics

Usage:
    python -m models.prediction.train_gru
"""
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
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
    logger.info(f"  Train: {data['X_lstm_train'].shape}")
    logger.info(f"  Val:   {data['X_lstm_val'].shape}")
    logger.info(f"  Test:  {data['X_lstm_test'].shape}")

    return {
        'X_train': data['X_lstm_train'],
        'y_train': data['y_lstm_train'],
        'X_val': data['X_lstm_val'],
        'y_val': data['y_lstm_val'],
        'X_test': data['X_lstm_test'],
        'y_test': data['y_lstm_test']
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


def build_gru_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    units_layer1: int = 128,
    units_layer2: int = 64,
    units_layer3: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> Any:
    """Build GRU model with configurable hyperparameters.

    Args:
        input_shape: (timesteps, features)
        num_classes: Number of output classes
        units_layer1: Units in first GRU layer
        units_layer2: Units in second GRU layer
        units_layer3: Units in third GRU layer
        dropout_rate: Dropout rate for regularization
        learning_rate: Initial learning rate

    Returns:
        Compiled Keras model
    """
    import keras
    from keras import layers

    model = keras.Sequential([
        # First GRU layer with return sequences
        layers.GRU(
            units_layer1,
            return_sequences=True,
            input_shape=input_shape,
            name='gru_1'
        ),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # Second GRU layer with return sequences
        layers.GRU(units_layer2, return_sequences=True, name='gru_2'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # Third GRU layer (final, no return sequences)
        layers.GRU(units_layer3, name='gru_3'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # Dense layers
        layers.Dense(16, activation='relu', name='dense_1'),
        layers.Dropout(dropout_rate / 2),

        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])

    # Learning rate schedule with cosine decay and restarts
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        learning_rate,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_gru_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hyperparameters: Dict[str, Any] = None,
    save_dir: str = "models/prediction/saved_models/gru"
) -> Dict[str, Any]:
    """Train GRU model with best practices.

    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        hyperparameters: Optional hyperparameters dict
        save_dir: Directory to save model and results

    Returns:
        Dictionary with training results and metrics
    """
    import keras

    logger.info("\n" + "=" * 60)
    logger.info("Training GRU Model")
    logger.info("=" * 60)

    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            'units_layer1': 128,
            'units_layer2': 64,
            'units_layer3': 32,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'max_epochs': 100,
            'early_stopping_patience': 5
        }

    logger.info("\nHyperparameters:")
    for key, value in hyperparameters.items():
        logger.info(f"  {key}: {value}")

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))

    logger.info(f"\nInput shape: {input_shape}")
    logger.info(f"Number of classes: {num_classes}")

    model = build_gru_model(
        input_shape=input_shape,
        num_classes=num_classes,
        units_layer1=hyperparameters['units_layer1'],
        units_layer2=hyperparameters['units_layer2'],
        units_layer3=hyperparameters['units_layer3'],
        dropout_rate=hyperparameters['dropout_rate'],
        learning_rate=hyperparameters['learning_rate']
    )

    logger.info("\nModel architecture:")
    model.summary(print_fn=logger.info)

    # Setup callbacks
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=hyperparameters['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),

        # Model checkpoint (save best model)
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_path / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),

        # CSV logger
        keras.callbacks.CSVLogger(
            str(save_path / 'training_history.csv'),
            append=False
        )
    ]

    # Train model
    logger.info("\nTraining...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=hyperparameters['max_epochs'],
        batch_size=hyperparameters['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    # Load best model
    logger.info("\nLoading best model...")
    best_model = keras.models.load_model(str(save_path / 'best_model.keras'))

    # Evaluate on all sets
    logger.info("\nEvaluating on all datasets...")
    train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)

    # Get predictions
    y_val_pred = np.argmax(best_model.predict(X_val, verbose=0), axis=1)

    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix

    class_names = ['down', 'neutral', 'up']
    class_report = classification_report(y_val, y_val_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_val, y_val_pred)

    # Financial metrics
    financial_metrics = calculate_financial_metrics(y_val, y_val_pred)

    # Results
    results = {
        'hyperparameters': hyperparameters,
        'training_time_hours': training_time / 3600,
        'epochs_completed': len(history.history['loss']),
        'train_accuracy': float(train_acc),
        'train_loss': float(train_loss),
        'val_accuracy': float(val_acc),
        'val_loss': float(val_loss),
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'financial_metrics': financial_metrics
    }

    # Save results
    results_file = save_path / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Training complete:")
    logger.info(f"  Training time: {training_time/3600:.2f} hours")
    logger.info(f"  Epochs: {len(history.history['loss'])}")
    logger.info(f"  Train accuracy: {train_acc:.4f}")
    logger.info(f"  Val accuracy: {val_acc:.4f}")

    logger.info(f"\n✓ Classification Report:")
    logger.info(classification_report(y_val, y_val_pred, target_names=class_names))

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

    logger.info(f"\n✓ Model saved to: {save_path / 'best_model.keras'}")
    logger.info(f"✓ Results saved to: {results_file}")

    return results


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("GRU Model Training Pipeline")
    logger.info("=" * 60)

    try:
        # Load data
        data = load_training_data()

        # Train model
        results = train_gru_model(
            data['X_train'],
            data['y_train'],
            data['X_val'],
            data['y_val']
        )

        logger.info("\n" + "=" * 60)
        logger.info("✓ GRU training complete!")
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

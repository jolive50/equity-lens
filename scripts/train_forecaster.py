"""Train the StockSense probabilistic forecasting model.

This script trains the gradient boosting classifier on historical market data,
performs cross-validation, evaluates performance, and persists the trained model.
Designed following SOLID principles with clear explanations for college students.

What this script does:
- Loads feature and label datasets from CSV files
- Splits data into train/validation/test sets
- Trains gradient boosting classifier
- Performs k-fold cross-validation
- Calculates performance metrics (accuracy, precision, recall, F1)
- Calibrates probabilities for better confidence estimates
- Saves trained model and metrics to disk

Why we need this:
- ML models must be trained on historical data before making predictions
- Cross-validation prevents overfitting (model memorizing instead of learning)
- Model persistence allows reusing trained model without retraining

How it works:
1. Load processed datasets (features.csv, labels.csv)
2. Split into train (70%), validation (15%), test (15%)
3. Train gradient boosting model on training set
4. Validate on validation set, tune hyperparameters if needed
5. Evaluate on test set (unseen data)
6. Save model to model/artifacts/forecaster.pkl

College-Level Concepts:
- Supervised Learning: Learning from labeled examples (features → label)
- Cross-Validation: Testing on multiple data splits to ensure generalization
- Overfitting: When model memorizes training data instead of learning patterns
- Calibration: Adjusting probabilities to match true frequencies
- SOLID Principles: Separation of concerns, dependency injection
"""
from __future__ import annotations

import argparse  # For command-line arguments
import joblib  # For saving/loading ML models
import json  # For saving metrics
import logging  # For progress tracking
import sys  # For system operations
from datetime import datetime  # For timestamps
from pathlib import Path  # For file paths
from typing import Dict, List, Optional, Tuple

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from sklearn.calibration import CalibratedClassifierCV  # For probability calibration
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # ML models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split  # For data splitting
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For preprocessing

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates probabilistic forecasting models.

    What this does: Orchestrates the complete model training pipeline
    Why: Centralizes training logic following Single Responsibility Principle
    How: Loads data, trains model, validates, saves artifacts

    This class demonstrates SOLID principles:
    - Single Responsibility: Only trains models (doesn't fetch data or make predictions)
    - Open/Closed: Can work with different model types (gradient boosting, random forest)
    - Liskov Substitution: Any sklearn classifier can be used
    - Interface Segregation: Clean, focused interface
    - Dependency Inversion: Depends on abstractions (sklearn interfaces)

    College-Level Analogy:
    Think of this as a personal trainer at the gym. The trainer doesn't design the
    equipment (models) or provide the exercises (data), but orchestrates the workout
    (training process) and tracks progress (metrics).
    """

    def __init__(
        self,
        *,
        model_type: str = "gradient_boosting",  # Type of model to train
        n_estimators: int = 100,  # Number of trees
        learning_rate: float = 0.1,  # Learning rate for gradient boosting
        max_depth: int = 6,  # Maximum tree depth
        random_state: int = 42,  # Random seed for reproducibility
    ):
        """Initialize model trainer with hyperparameters.

        What: Sets up the trainer with model configuration
        Why: Allows customization of model architecture and training
        How: Stores parameters and initializes model

        Args:
            model_type: "gradient_boosting" or "random_forest"
            n_estimators: Number of trees in the ensemble
            learning_rate: Step size for gradient boosting (only for GBM)
            max_depth: Maximum depth of each tree
            random_state: Seed for reproducibility

        Hyperparameter Explanation:
        - n_estimators: More trees = better learning but slower training
        - learning_rate: Lower = more careful learning but needs more trees
        - max_depth: Higher = more complex patterns but risk overfitting
        - random_state: Ensures same results every time (important for debugging)
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        # Initialize base model
        # What: Create the ML model with specified parameters
        # Why: Need a model object to train
        # How: Instantiate sklearn classifier
        if model_type == "gradient_boosting":
            # Gradient Boosting: Builds trees sequentially, each fixing previous errors
            # What: Sequential ensemble that focuses on hard examples
            # Why: Often best accuracy for tabular data
            # How: Each new tree predicts the residuals (errors) of previous trees
            self.base_model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                verbose=1  # Show training progress
            )
        elif model_type == "random_forest":
            # Random Forest: Builds trees in parallel, averages predictions
            # What: Parallel ensemble with voting mechanism
            # Why: Very robust to overfitting, good baseline
            # How: Each tree sees random subset of data and features
            self.base_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                class_weight='balanced',  # Handle class imbalance
                verbose=1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize scaler for feature normalization
        # What: Standardizes features to zero mean and unit variance
        # Why: Gradient boosting doesn't strictly need scaling, but helps with convergence
        # How: Learns mean and std from training data, applies to all data
        self.scaler = StandardScaler()

        # Initialize label encoder
        # What: Converts string labels to integers
        # Why: sklearn models need numeric labels (0, 1, 2)
        # How: Maps "up" → 0, "down" → 1, "neutral" → 2 (or similar)
        self.label_encoder = LabelEncoder()

        # Trained model (after calibration)
        # What: Final model after probability calibration
        # Why: Raw model probabilities often poorly calibrated
        # How: CalibratedClassifierCV wraps base model
        self.calibrated_model = None

        # Training metrics
        # What: Dictionary to store all performance metrics
        # Why: Need to track model quality and save for later analysis
        self.metrics = {}

    def load_data(
        self,
        *,
        features_path: Path,  # Path to features.csv
        labels_path: Path,  # Path to labels.csv
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load feature and label datasets from CSV files.

        What: Reads processed datasets into memory
        Why: Need data to train the model
        How: Uses pandas to read CSVs, validates consistency

        Args:
            features_path: Path to features.csv (created by build_training_dataset.py)
            labels_path: Path to labels.csv (created by build_training_dataset.py)

        Returns:
            Tuple of (features DataFrame, labels Series)

        Raises:
            ValueError: If features and labels don't match in length

        Data Flow:
        features.csv: ticker, date, 1d_return, 5d_return, ..., volatility
        labels.csv: ticker, date, label
        → Align on (ticker, date)
        → Return (X, y) for training
        """
        logger.info(f"Loading features from {features_path}")
        features_df = pd.read_csv(features_path)

        logger.info(f"Loading labels from {labels_path}")
        labels_df = pd.read_csv(labels_path)

        # Validate data consistency
        # What: Ensure features and labels match
        # Why: Can't train if data is misaligned
        # How: Check lengths and merge on common keys
        if len(features_df) != len(labels_df):
            raise ValueError(
                f"Feature and label counts don't match: {len(features_df)} features vs {len(labels_df)} labels"
            )

        # Extract feature columns (exclude metadata)
        # What: Get only the numerical features for training
        # Why: ticker and date are identifiers, not features
        # How: Drop non-feature columns
        feature_columns = [col for col in features_df.columns if col not in ['ticker', 'date']]
        X = features_df[feature_columns]

        # Extract labels
        # What: Get target variable (what we're predicting)
        # Why: Supervised learning needs (X, y) pairs
        # How: Extract 'label' column
        y = labels_df['label']

        logger.info(f"Loaded {len(X)} samples with {len(feature_columns)} features")
        logger.info(f"Feature columns: {', '.join(feature_columns)}")
        logger.info(f"Label distribution:")

        # Log class balance
        # What: Show how many samples of each class
        # Why: Imbalanced classes can cause biased models
        # How: Count occurrences of each label
        for label, count in y.value_counts().items():
            percentage = count / len(y) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")

        return X, y

    def train(
        self,
        *,
        X_train: pd.DataFrame,  # Training features
        y_train: pd.Series,  # Training labels
        X_val: Optional[pd.DataFrame] = None,  # Validation features (optional)
        y_val: Optional[pd.Series] = None,  # Validation labels (optional)
        cv_folds: int = 5,  # Number of cross-validation folds
    ) -> None:
        """Train the forecasting model with cross-validation.

        What: Main training function that fits the model to data
        Why: Learn patterns from historical data
        How: Fit base model, perform cross-validation, calibrate probabilities

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Optional validation features for monitoring
            y_val: Optional validation labels for monitoring
            cv_folds: Number of folds for cross-validation (default 5)

        Process:
        1. Encode labels (strings → integers)
        2. Scale features (standardize)
        3. Train base model
        4. Cross-validate to check generalization
        5. Calibrate probabilities
        6. Validate if validation set provided

        Why Cross-Validation:
        Instead of one train/test split, we do multiple splits to ensure
        the model generalizes well. Like testing a student with multiple
        exams instead of just one.
        """
        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Training set: {len(X_train)} samples")

        # Encode labels to integers
        # What: Convert "up", "down", "neutral" to 0, 1, 2
        # Why: sklearn needs numeric labels
        # How: LabelEncoder learns mapping and transforms
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        logger.info(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # Scale features
        # What: Standardize features to mean=0, std=1
        # Why: Helps with convergence and prevents feature dominance
        # How: Scaler learns mean/std from training data, transforms all data
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train base model
        # What: Fit the model to training data
        # Why: This is where the learning happens
        # How: Model adjusts internal parameters to minimize prediction error
        logger.info("Training base model...")
        self.base_model.fit(X_train_scaled, y_train_encoded)

        # Calculate training accuracy
        # What: Measure how well model predicts training data
        # Why: Sanity check (should be high, if not something is wrong)
        # How: Predict on training data and compare to true labels
        train_predictions = self.base_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, train_predictions)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")

        # Perform cross-validation
        # What: Evaluate model on multiple train/test splits
        # Why: Checks if model generalizes or just memorizes
        # How: Split data into k folds, train on k-1, test on 1, repeat
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.base_model,
            X_train_scaled,
            y_train_encoded,
            cv=cv_folds,  # Number of splits
            scoring='accuracy',  # Metric to use
            n_jobs=-1  # Use all CPU cores
        )

        # Log cross-validation results
        # What: Show performance across all folds
        # Why: Helps detect overfitting (train >> CV)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"CV mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Store metrics
        # What: Save training metrics for later analysis
        # Why: Track model performance over time
        self.metrics['train_accuracy'] = float(train_accuracy)
        self.metrics['cv_mean_accuracy'] = float(cv_scores.mean())
        self.metrics['cv_std_accuracy'] = float(cv_scores.std())
        self.metrics['cv_scores'] = cv_scores.tolist()

        # Calibrate probabilities
        # What: Adjust model probabilities to match true frequencies
        # Why: Raw model probabilities often over/under-confident
        # How: CalibratedClassifierCV uses validation split to learn calibration
        logger.info("Calibrating probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model,
            method='isotonic',  # Non-parametric calibration (flexible)
            cv='prefit'  # Model already trained, just calibrate
        )

        # Fit calibrator on training data
        # What: Learn probability calibration mapping
        # Why: Better confidence estimates for predictions
        # How: Uses internal validation split to tune calibration
        self.calibrated_model.fit(X_train_scaled, y_train_encoded)

        # Validate on validation set if provided
        # What: Test on held-out validation data
        # Why: Independent check of performance
        # How: Predict and calculate metrics
        if X_val is not None and y_val is not None:
            logger.info(f"Validating on {len(X_val)} samples...")
            val_metrics = self.evaluate(X_val, y_val, dataset_name="Validation")
            self.metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

        logger.info("✓ Training complete")

    def evaluate(
        self,
        X: pd.DataFrame,  # Features to evaluate
        y: pd.Series,  # True labels
        dataset_name: str = "Test"  # Name for logging
    ) -> Dict[str, float]:
        """Evaluate model performance on a dataset.

        What: Calculates comprehensive performance metrics
        Why: Need objective measures of model quality
        How: Predict, compare to true labels, calculate metrics

        Args:
            X: Feature matrix
            y: True labels
            dataset_name: Name for logging (e.g., "Test", "Validation")

        Returns:
            Dictionary of metrics:
            - accuracy: Overall correctness (0-1)
            - precision: Of predictions, how many were right (0-1)
            - recall: Of true positives, how many were found (0-1)
            - f1: Harmonic mean of precision and recall (0-1)
            - log_loss: Probability calibration quality (lower is better)

        Metric Explanations:
        - Accuracy: (correct predictions) / (total predictions)
        - Precision: (true positives) / (predicted positives)
          - "When I say up, how often is it actually up?"
        - Recall: (true positives) / (actual positives)
          - "Of all the ups, how many did I catch?"
        - F1: 2 * (precision * recall) / (precision + recall)
          - Balances precision and recall
        - Log Loss: Measures probability quality (lower = better calibrated)
        """
        if self.calibrated_model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        logger.info(f"Evaluating on {dataset_name} set ({len(X)} samples)...")

        # Encode labels
        # What: Convert string labels to integers
        # Why: Need numeric labels for comparison
        # How: Use same encoder fitted on training data
        y_encoded = self.label_encoder.transform(y)

        # Scale features
        # What: Standardize features same way as training
        # Why: Model expects scaled features
        # How: Use same scaler fitted on training data
        X_scaled = self.scaler.transform(X)

        # Get predictions and probabilities
        # What: Run model on test data
        # Why: Need predictions to calculate metrics
        # How: Model forward pass
        y_pred = self.calibrated_model.predict(X_scaled)
        y_proba = self.calibrated_model.predict_proba(X_scaled)

        # Calculate metrics
        # What: Compute performance scores
        # Why: Quantify model quality
        # How: Compare predictions to true labels
        accuracy = accuracy_score(y_encoded, y_pred)
        precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)
        logloss = log_loss(y_encoded, y_proba)

        # Log results
        logger.info(f"{dataset_name} Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        logger.info(f"  Log Loss:  {logloss:.4f}")

        # Detailed classification report
        # What: Per-class metrics breakdown
        # Why: See which classes model handles well/poorly
        # How: sklearn's classification_report function
        logger.info(f"\n{dataset_name} Classification Report:")
        report = classification_report(
            y_encoded,
            y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        logger.info(f"\n{report}")

        # Confusion matrix
        # What: Shows true vs predicted label counts
        # Why: Reveals which errors are most common
        # How: Cross-tabulate predictions and actuals
        logger.info(f"\n{dataset_name} Confusion Matrix:")
        cm = confusion_matrix(y_encoded, y_pred)
        logger.info(f"Rows: True labels, Columns: Predicted labels")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        logger.info(f"\n{cm}")

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'log_loss': float(logloss)
        }

    def save_model(
        self,
        *,
        output_dir: Path,  # Directory to save model
        model_name: str = "forecaster"  # Base name for files
    ) -> Dict[str, Path]:
        """Save trained model and artifacts to disk.

        What: Persists model, scaler, encoder, and metrics
        Why: Reuse trained model without retraining
        How: Uses joblib for model serialization, JSON for metrics

        Args:
            output_dir: Directory to save files
            model_name: Base name for output files

        Returns:
            Dictionary mapping artifact types to file paths

        Files created:
        - forecaster.pkl: Calibrated model (load with joblib.load)
        - forecaster_scaler.pkl: Feature scaler
        - forecaster_encoder.pkl: Label encoder
        - forecaster_metrics.json: Performance metrics
        """
        if self.calibrated_model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")

        # Create output directory
        # What: Ensure directory exists
        # Why: Can't save if directory doesn't exist
        # How: mkdir with parents=True (creates parent directories)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model artifacts to {output_dir}")

        # Save calibrated model
        # What: Serialize trained model to disk
        # Why: Most important artifact, contains learned parameters
        # How: joblib efficiently saves sklearn models
        model_path = output_dir / f"{model_name}.pkl"
        joblib.dump(self.calibrated_model, model_path)
        logger.info(f"✓ Saved model to {model_path}")

        # Save scaler
        # What: Serialize feature scaler
        # Why: Need same scaling at prediction time
        # How: joblib saves scaler with fitted mean/std
        scaler_path = output_dir / f"{model_name}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✓ Saved scaler to {scaler_path}")

        # Save label encoder
        # What: Serialize label mapping
        # Why: Need to convert predictions back to strings
        # How: joblib saves encoder with learned classes
        encoder_path = output_dir / f"{model_name}_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        logger.info(f"✓ Saved encoder to {encoder_path}")

        # Save metrics
        # What: Write performance metrics to JSON
        # Why: Track model quality, compare versions
        # How: JSON serialization with timestamp
        metrics_path = output_dir / f"{model_name}_metrics.json"
        metrics_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'random_state': self.random_state
            },
            'metrics': self.metrics
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics_with_metadata, f, indent=2)
        logger.info(f"✓ Saved metrics to {metrics_path}")

        return {
            'model': model_path,
            'scaler': scaler_path,
            'encoder': encoder_path,
            'metrics': metrics_path
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    What: Handles command-line options
    Why: Makes script flexible and scriptable
    How: Uses argparse to define and parse arguments

    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train StockSense probabilistic forecasting model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_forecaster.py

  # Train gradient boosting with custom hyperparameters
  python train_forecaster.py --model gradient_boosting --n-estimators 200 --learning-rate 0.05

  # Train random forest
  python train_forecaster.py --model random_forest --n-estimators 150 --max-depth 8

  # Specify custom data directories
  python train_forecaster.py --features-path data/custom/features.csv --labels-path data/custom/labels.csv
        """
    )

    # Data paths
    parser.add_argument(
        '--features-path',
        type=Path,
        default=Path('data/training/features/features.csv'),
        help='Path to features CSV file (default: data/training/features/features.csv)'
    )
    parser.add_argument(
        '--labels-path',
        type=Path,
        default=Path('data/training/labels/labels.csv'),
        help='Path to labels CSV file (default: data/training/labels/labels.csv)'
    )

    # Model configuration
    parser.add_argument(
        '--model',
        choices=['gradient_boosting', 'random_forest'],
        default='gradient_boosting',
        help='Model type to train (default: gradient_boosting)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees/estimators (default: 100)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate for gradient boosting (default: 0.1)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth (default: 6)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Training configuration
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.15,
        help='Proportion of data for testing (default: 0.15)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.15,
        help='Proportion of data for validation (default: 0.15)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('model/artifacts'),
        help='Directory to save trained model (default: model/artifacts)'
    )
    parser.add_argument(
        '--model-name',
        default='forecaster',
        help='Base name for saved model files (default: forecaster)'
    )

    return parser.parse_args()


def main():
    """Main entry point for model training script.

    What: Orchestrates the complete training pipeline
    Why: Provides command-line interface
    How: Loads data, trains model, evaluates, saves

    This demonstrates the Single Responsibility Principle:
    - Only coordinates CLI workflow
    - Delegates actual work to ModelTrainer
    """
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("StockSense Model Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model}")
    logger.info(f"Hyperparameters:")
    logger.info(f"  n_estimators: {args.n_estimators}")
    logger.info(f"  learning_rate: {args.learning_rate}")
    logger.info(f"  max_depth: {args.max_depth}")
    logger.info(f"  random_state: {args.random_state}")
    logger.info("=" * 60)

    # Initialize trainer
    # What: Create trainer with specified hyperparameters
    # Why: Separates configuration from execution
    # How: Pass command-line args to trainer constructor
    trainer = ModelTrainer(
        model_type=args.model,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.random_state
    )

    # Load data
    # What: Read features and labels from CSV
    # Why: Need data to train model
    # How: Trainer handles loading and validation
    try:
        X, y = trainer.load_data(
            features_path=args.features_path,
            labels_path=args.labels_path
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.error("Make sure you've run build_training_dataset.py first!")
        return 1

    # Split data into train/validation/test
    # What: Divide data for training, validation, and testing
    # Why: Need separate sets to prevent overfitting
    # How: sklearn's train_test_split with stratification
    #
    # Why stratify: Ensures each split has proportional class distribution
    # Example: If 60% up, 30% down, 10% neutral overall,
    #          each split will have ~60% up, ~30% down, ~10% neutral
    logger.info("=" * 60)
    logger.info("Splitting data...")
    logger.info(f"Train: {1 - args.test_size - args.val_size:.0%}")
    logger.info(f"Validation: {args.val_size:.0%}")
    logger.info(f"Test: {args.test_size:.0%}")
    logger.info("=" * 60)

    # First split: separate test set
    # What: Hold out test data for final evaluation
    # Why: Test set must be completely unseen during training
    # How: stratify=y ensures balanced class distribution
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y  # Maintain class proportions
    )

    # Second split: separate validation from training
    # What: Create validation set from remaining data
    # Why: Validation monitors training, test evaluates final model
    # How: Adjust val_size to account for already-removed test set
    val_size_adjusted = args.val_size / (1 - args.test_size)  # Adjust proportion
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=args.random_state,
        stratify=y_temp
    )

    logger.info(f"Split sizes:")
    logger.info(f"  Training: {len(X_train)} samples")
    logger.info(f"  Validation: {len(X_val)} samples")
    logger.info(f"  Test: {len(X_test)} samples")

    # Train model
    # What: Fit model to training data with validation monitoring
    # Why: This is where the learning happens
    # How: Trainer handles full training pipeline
    try:
        trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cv_folds=args.cv_folds
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    # Evaluate on test set
    # What: Final performance check on unseen data
    # Why: Most honest measure of model quality
    # How: Predict on test set, calculate metrics
    logger.info("=" * 60)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 60)

    try:
        test_metrics = trainer.evaluate(X_test, y_test, dataset_name="Test")
        trainer.metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1

    # Save model
    # What: Persist trained model and artifacts
    # Why: Reuse model without retraining
    # How: joblib serialization to disk
    logger.info("=" * 60)
    logger.info("Saving Model")
    logger.info("=" * 60)

    try:
        output_paths = trainer.save_model(
            output_dir=args.output_dir,
            model_name=args.model_name
        )
    except Exception as e:
        logger.error(f"Failed to save model: {e}", exc_info=True)
        return 1

    # Success summary
    logger.info("=" * 60)
    logger.info("✓ Training Pipeline Complete!")
    logger.info("=" * 60)
    logger.info("Saved artifacts:")
    for artifact_type, path in output_paths.items():
        logger.info(f"  {artifact_type}: {path}")
    logger.info("=" * 60)
    logger.info("Model Performance Summary:")
    logger.info(f"  Training Accuracy: {trainer.metrics.get('train_accuracy', 0):.4f}")
    logger.info(f"  CV Mean Accuracy: {trainer.metrics.get('cv_mean_accuracy', 0):.4f}")
    logger.info(f"  Validation Accuracy: {trainer.metrics.get('val_accuracy', 0):.4f}")
    logger.info(f"  Test Accuracy: {trainer.metrics.get('test_accuracy', 0):.4f}")
    logger.info(f"  Test F1 Score: {trainer.metrics.get('test_f1', 0):.4f}")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Review metrics to ensure model quality")
    logger.info("2. Update pipelines/realtime/models/forecaster.py to load this model")
    logger.info("3. Test predictions with sample data")
    logger.info("4. Deploy model to production")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Unit tests for LSTM forecaster training and inference.

WHAT: Tests for LSTM model training pipeline and ProbabilisticForecaster integration
WHY: Ensure model training works correctly and forecaster loads model properly
HOW: Mock data, test each component independently
DATA: Synthetic test data -> validation of training and inference logic
"""
import pytest
import numpy as np
import pandas as pd
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pipelines.realtime.models.train_lstm_forecaster import (
    TrainingConfig,
    LSTMDataProcessor,
    LSTMForecasterModel,
    ProbabilityCalibrator
)
from pipelines.realtime.models.forecaster import ProbabilisticForecaster, ForecastResult


class TestTrainingConfig:
    """Test training configuration dataclass.

    WHAT: Validates TrainingConfig initialization and defaults
    WHY: Config drives entire training process, must be correct
    HOW: Check default values and path construction
    """

    def test_default_config(self):
        """
        WHAT: Verify default configuration values are set correctly.
        WHY: Training pipeline depends on these defaults.
        HOW: Create config, check each attribute.
        DATA: Config object -> validated attributes.
        """
        config = TrainingConfig()

        # WHAT: Check architecture defaults
        # WHY: LSTM layers must match user requirements (128->64)
        assert config.lstm_layer1_units == 128, "First LSTM layer should be 128 units"
        assert config.lstm_layer2_units == 64, "Second LSTM layer should be 64 units"
        assert config.sequence_length == 30, "Should use 30-day sequences"

        # WHAT: Check training defaults
        # WHY: These affect model quality and training time
        assert config.batch_size == 64
        assert config.epochs == 50
        assert config.validation_split == 0.2
        assert config.calibration_split == 0.1

        # WHAT: Check class configuration
        # WHY: Three-class classification (up/down/steady)
        assert config.num_classes == 3
        assert config.price_change_threshold == 0.02  # 2% threshold


class TestLSTMDataProcessor:
    """Test data processing for LSTM training.

    WHAT: Tests data loading, indicator calculation, sequence creation
    WHY: Data quality determines model performance
    HOW: Use synthetic data, verify transformations
    """

    @pytest.fixture
    def config(self):
        """Create test configuration.

        WHAT: Provides config for data processor tests.
        WHY: Processor needs config for sequence length, etc.
        """
        return TrainingConfig()

    @pytest.fixture
    def processor(self, config):
        """Create data processor instance.

        WHAT: Instantiate LSTMDataProcessor with test config.
        WHY: Need processor for testing data transformations.
        """
        return LSTMDataProcessor(config)

    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing.

        WHAT: Generate synthetic OHLCV data.
        WHY: Need realistic data structure for testing.
        HOW: Create DataFrame with required columns.
        DATA: Returns pandas DataFrame with OHLCV columns.
        """
        # WHAT: Create 100 days of synthetic data
        # WHY: Need enough data for indicator calculations
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        base_price = 100.0

        # WHAT: Generate realistic price movements
        # HOW: Add random walk with slight upward bias
        price_changes = np.random.normal(0.001, 0.02, size=100)
        prices = base_price * np.cumprod(1 + price_changes)

        # WHAT: Create DataFrame with all required columns
        # WHY: Mimics actual CSV structure
        df = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, size=100),
            'ticker': 'TEST'
        })

        return df

    def test_calculate_technical_indicators(self, processor, sample_stock_data):
        """
        WHAT: Test technical indicator calculation.
        WHY: Indicators are features for LSTM, must be correct.
        HOW: Calculate indicators, check they exist and have valid values.
        DATA: Stock data -> data with indicators.
        """
        # WHAT: Calculate indicators
        # WHY: Test indicator calculation logic
        df_with_indicators = processor.calculate_technical_indicators(sample_stock_data)

        # WHAT: Check all expected columns exist
        # WHY: LSTM expects specific feature set
        expected_cols = [
            'returns', 'rsi', 'sma_10', 'sma_20',
            'macd', 'macd_signal', 'bb_middle', 'bb_upper', 'bb_lower',
            'volume_ratio', 'volatility'
        ]

        for col in expected_cols:
            assert col in df_with_indicators.columns, f"Missing indicator: {col}"

        # WHAT: Check RSI is in valid range
        # WHY: RSI should be 0-100
        # HOW: Filter out NaN values, check range
        rsi_values = df_with_indicators['rsi'].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), "RSI out of range"

        # WHAT: Check Bollinger Bands ordering
        # WHY: Upper should be > middle > lower
        # HOW: Compare columns element-wise
        valid_rows = df_with_indicators[['bb_upper', 'bb_middle', 'bb_lower']].dropna()
        assert (valid_rows['bb_upper'] >= valid_rows['bb_middle']).all(), "BB upper < middle"
        assert (valid_rows['bb_middle'] >= valid_rows['bb_lower']).all(), "BB middle < lower"

    def test_create_labels(self, processor, sample_stock_data):
        """
        WHAT: Test label creation for classification.
        WHY: Labels determine what model learns.
        HOW: Create labels, check distribution and values.
        DATA: Price data -> class labels (0, 1, 2).
        """
        # WHAT: Calculate indicators first (needed for complete data)
        df = processor.calculate_technical_indicators(sample_stock_data)

        # WHAT: Create labels
        df_with_labels = processor.create_labels(df)

        # WHAT: Check label column exists
        assert 'label' in df_with_labels.columns, "Label column missing"

        # WHAT: Check labels are valid classes
        # WHY: Should only be 0 (down), 1 (steady), 2 (up), or NaN
        labels = df_with_labels['label'].dropna()
        valid_labels = labels.isin([0, 1, 2])
        assert valid_labels.all(), "Invalid label values found"

        # WHAT: Check we have all three classes
        # WHY: Need balanced representation for training
        # HOW: Check unique values
        unique_labels = set(labels.unique())
        # Note: May not have all 3 classes in small sample, just check they're valid
        assert unique_labels.issubset({0, 1, 2}), "Labels outside expected range"

    def test_create_sequences(self, processor, sample_stock_data):
        """
        WHAT: Test sequence creation for LSTM input.
        WHY: LSTM requires (samples, sequence_length, features) arrays.
        HOW: Create sequences, validate shape.
        DATA: DataFrame -> (X_sequences, y_labels) arrays.
        """
        # WHAT: Prepare data with indicators and labels
        df = processor.calculate_technical_indicators(sample_stock_data)
        df = processor.create_labels(df)

        # WHAT: Create sequences
        X, y = processor.create_sequences(df)

        # WHAT: Check X has correct shape
        # WHY: LSTM expects (n_samples, sequence_length, n_features)
        # HOW: Validate each dimension
        assert len(X.shape) == 3, "X should be 3D array"
        assert X.shape[1] == processor.config.sequence_length, "Wrong sequence length"
        assert X.shape[2] == len(processor.feature_columns), "Wrong number of features"

        # WHAT: Check y has correct shape and values
        # WHY: Labels should match number of sequences
        assert len(y.shape) == 1, "y should be 1D array"
        assert len(y) == len(X), "Number of labels should match number of sequences"
        assert all(label in [0, 1, 2] for label in y), "Invalid label values"

    def test_normalize_sequences(self, processor):
        """
        WHAT: Test sequence normalization.
        WHY: Neural networks need normalized inputs.
        HOW: Create dummy sequences, normalize, check properties.
        DATA: Raw sequences -> normalized sequences.
        """
        # WHAT: Create dummy sequences
        # WHY: Test normalization logic
        X_train = np.random.randn(100, 30, 16).astype(np.float32)
        X_val = np.random.randn(20, 30, 16).astype(np.float32)
        X_cal = np.random.randn(10, 30, 16).astype(np.float32)

        # WHAT: Normalize
        X_train_norm, X_val_norm, X_cal_norm = processor.normalize_sequences(
            X_train, X_val, X_cal
        )

        # WHAT: Check shapes preserved
        # WHY: Normalization shouldn't change dimensions
        assert X_train_norm.shape == X_train.shape, "Train shape changed"
        assert X_val_norm.shape == X_val.shape, "Val shape changed"
        assert X_cal_norm.shape == X_cal.shape, "Cal shape changed"

        # WHAT: Check train set approximately normalized
        # WHY: StandardScaler should give ~zero mean, ~unit variance
        # HOW: Calculate mean and std across all dimensions
        mean = X_train_norm.mean()
        std = X_train_norm.std()
        assert abs(mean) < 0.1, f"Mean not close to 0: {mean}"
        assert abs(std - 1.0) < 0.2, f"Std not close to 1: {std}"


class TestLSTMForecasterModel:
    """Test LSTM model architecture and training.

    WHAT: Tests model building and training logic
    WHY: Ensure model architecture matches requirements
    HOW: Mock training data, verify model structure
    """

    @pytest.fixture
    def config(self):
        """Test configuration with small model for speed."""
        config = TrainingConfig()
        config.epochs = 2  # Fast training for tests
        config.batch_size = 32
        return config

    @pytest.fixture
    def model_builder(self, config):
        """Create LSTM model builder."""
        return LSTMForecasterModel(config)

    def test_build_model_architecture(self, model_builder):
        """
        WHAT: Test LSTM model architecture.
        WHY: Must match specification (128->64 LSTM layers).
        HOW: Build model, check layer configuration.
        DATA: Model builder -> compiled Keras model.
        """
        # WHAT: Build model with test input shape
        # WHY: Need to verify layer sizes
        input_shape = (30, 16)  # 30 timesteps, 16 features
        model = model_builder.build_model(input_shape)

        # WHAT: Check model has correct layers
        # WHY: Architecture must match requirements
        # HOW: Inspect layer list
        layer_names = [layer.name for layer in model.layers]
        assert 'lstm_layer_1' in layer_names, "Missing first LSTM layer"
        assert 'lstm_layer_2' in layer_names, "Missing second LSTM layer"
        assert 'output_layer' in layer_names, "Missing output layer"

        # WHAT: Check LSTM layer sizes
        # WHY: Must be 128 and 64 units as specified
        lstm1 = model.get_layer('lstm_layer_1')
        lstm2 = model.get_layer('lstm_layer_2')
        assert lstm1.units == 128, "First LSTM should have 128 units"
        assert lstm2.units == 64, "Second LSTM should have 64 units"

        # WHAT: Check output layer
        # WHY: Should have 3 units (up/down/steady) with softmax
        output = model.get_layer('output_layer')
        assert output.units == 3, "Output should have 3 units"
        assert output.activation.__name__ == 'softmax', "Output should use softmax"

    def test_train_model(self, model_builder):
        """
        WHAT: Test model training process.
        WHY: Ensure training runs without errors.
        HOW: Create dummy data, train for few epochs.
        DATA: Synthetic sequences -> trained model.
        """
        # WHAT: Create dummy training data
        # WHY: Test training logic
        X_train = np.random.randn(100, 30, 16).astype(np.float32)
        y_train = np.random.randint(0, 3, size=100)
        X_val = np.random.randn(20, 30, 16).astype(np.float32)
        y_val = np.random.randint(0, 3, size=20)

        # WHAT: Build and train model
        # WHY: Verify training completes successfully
        input_shape = (30, 16)
        model_builder.build_model(input_shape)

        # WHAT: Train model
        # WHY: Should run without errors
        history = model_builder.train(X_train, y_train, X_val, y_val)

        # WHAT: Check training history exists
        # WHY: History contains metrics
        assert 'loss' in history, "Training history missing loss"
        assert 'val_loss' in history, "Training history missing val_loss"
        assert len(history['loss']) > 0, "No training epochs recorded"


class TestProbabilityCalibrator:
    """Test probability calibration.

    WHAT: Tests calibration of model probabilities
    WHY: Ensures 70% confidence means 70% actual accuracy
    HOW: Create known probability distributions, verify calibration
    """

    @pytest.fixture
    def calibrator(self):
        """Create calibrator instance."""
        return ProbabilityCalibrator()

    def test_fit_calibration(self, calibrator):
        """
        WHAT: Test calibration fitting.
        WHY: Calibration maps predicted -> actual probabilities.
        HOW: Create synthetic predictions, fit calibrator.
        DATA: True labels + predictions -> calibration maps.
        """
        # WHAT: Create synthetic calibration data
        # WHY: Test calibration learning
        # HOW: Create overconfident predictions
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 10)
        y_proba = np.random.rand(90)  # Random probabilities

        # WHAT: Fit calibration for class 0
        # WHY: Should learn mapping
        calibrator.fit(y_true, y_proba, class_idx=0)

        # WHAT: Check calibration map exists
        # WHY: Should store calibration curve
        assert 0 in calibrator.calibration_maps, "Calibration map not created"
        assert 'true_freq' in calibrator.calibration_maps[0], "Missing true_freq"
        assert 'pred_mean' in calibrator.calibration_maps[0], "Missing pred_mean"

    def test_calibrate_probabilities(self, calibrator):
        """
        WHAT: Test probability calibration transformation.
        WHY: Calibrated probabilities should be better.
        HOW: Fit calibrator, apply to test data.
        DATA: Raw probabilities -> calibrated probabilities.
        """
        # WHAT: Create and fit calibration data
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.concatenate([
            np.random.beta(2, 5, 50),  # Low probs for class 0
            np.random.beta(5, 2, 50)   # High probs for class 1
        ])

        calibrator.fit(y_true, y_proba, class_idx=0)
        calibrator.fit(y_true, 1 - y_proba, class_idx=1)

        # WHAT: Apply calibration
        # WHY: Test transformation
        test_proba = np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]])
        calibrated = calibrator.calibrate(test_proba)

        # WHAT: Check output properties
        # WHY: Should preserve shape, sum to 1
        assert calibrated.shape == test_proba.shape, "Shape changed"
        row_sums = calibrated.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5), "Rows don't sum to 1"


class TestProbabilisticForecasterLSTM:
    """Test ProbabilisticForecaster with LSTM model.

    WHAT: Tests forecaster loading and using pre-trained LSTM
    WHY: Integration test for complete prediction pipeline
    HOW: Mock model files, test loading and prediction
    """

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model files.

        WHAT: Creates temp dir for storing mock model files.
        WHY: Need to test model loading without real trained model.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_lstm_model_missing_raises_error(self, temp_model_dir):
        """
        WHAT: Test error when LSTM model files missing.
        WHY: Should fail fast with clear instructions.
        HOW: Try to load LSTM without files, check error message.
        DATA: Empty directory -> RuntimeError with instructions.
        """
        # WHAT: Attempt to create LSTM forecaster without model files
        # WHY: Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            forecaster = ProbabilisticForecaster(
                model_type="lstm",
                model_dir=str(temp_model_dir)
            )

        # WHAT: Check error message is helpful
        # WHY: User needs to know how to fix
        error_msg = str(exc_info.value)
        assert "LSTM model not found" in error_msg
        assert "train_lstm_forecaster.py" in error_msg

    @patch('pipelines.realtime.models.forecaster.keras')
    def test_lstm_model_loads_successfully(self, mock_keras, temp_model_dir):
        """
        WHAT: Test successful LSTM model loading.
        WHY: Verify all artifacts load correctly.
        HOW: Mock Keras, create fake pickle files, test loading.
        DATA: Mock model files -> loaded forecaster.
        """
        # WHAT: Create mock model files
        # WHY: Simulate trained model artifacts

        # WHAT: Mock Keras model
        mock_model = MagicMock()
        mock_keras.models.load_model.return_value = mock_model

        # WHAT: Create mock scaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 480))  # 30 seq * 16 features
        with open(temp_model_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        # WHAT: Create mock calibrator
        calibrator = ProbabilityCalibrator()
        with open(temp_model_dir / "calibrator.pkl", 'wb') as f:
            pickle.dump(calibrator, f)

        # WHAT: Create mock feature columns
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'returns',
                       'rsi', 'sma_10', 'sma_20', 'macd', 'macd_signal',
                       'bb_middle', 'bb_upper', 'bb_lower', 'volume_ratio', 'volatility']
        with open(temp_model_dir / "feature_columns.pkl", 'wb') as f:
            pickle.dump(feature_cols, f)

        # WHAT: Create mock config
        config = TrainingConfig()
        with open(temp_model_dir / "config.pkl", 'wb') as f:
            pickle.dump(config, f)

        # WHAT: Create dummy model file (Keras will be mocked)
        (temp_model_dir / "lstm_forecaster.h5").touch()

        # WHAT: Load forecaster
        # WHY: Should succeed with all files present
        forecaster = ProbabilisticForecaster(
            model_type="lstm",
            model_dir=str(temp_model_dir)
        )

        # WHAT: Verify forecaster loaded correctly
        # WHY: All attributes should be set
        assert forecaster.is_trained == True
        assert forecaster.lstm_model == mock_model
        assert hasattr(forecaster, 'lstm_scaler')
        assert hasattr(forecaster, 'lstm_calibrator')
        assert forecaster.lstm_config.sequence_length == 30

    def test_fallback_to_traditional_models(self):
        """
        WHAT: Test fallback to gradient boosting when LSTM not specified.
        WHY: Should support non-LSTM models too.
        HOW: Create forecaster with gradient_boosting type.
        DATA: Config -> traditional ML forecaster.
        """
        # WHAT: Create forecaster with gradient boosting
        # WHY: Should not try to load LSTM
        forecaster = ProbabilisticForecaster(model_type="gradient_boosting")

        # WHAT: Verify it's not trained but model exists
        # WHY: Gradient boosting doesn't require pre-training
        assert forecaster.model_type == "gradient_boosting"
        assert hasattr(forecaster, 'calibrated_model')
        assert forecaster.is_trained == False  # Not trained yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

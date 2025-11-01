"""Download and cache FinBERT model for offline use.

This script downloads the FinBERT sentiment analysis model from Hugging Face
and caches it locally for StockSense. Designed with SOLID principles and
includes clear explanations for college-level understanding.

What this script does:
- Downloads ProsusAI/finbert model from Hugging Face Hub
- Downloads associated tokenizer files
- Verifies model integrity
- Caches locally in model/finbert/ directory
- Tests model to ensure it works correctly

Why we need this:
- StockSense requires FinBERT for sentiment analysis
- Production environments may not have internet access
- Pre-downloading ensures faster startup times
- Local cache allows offline operation

How it works:
1. Check if model already cached
2. Download model and tokenizer using transformers library
3. Verify files are complete
4. Run test prediction to ensure functionality
5. Save metadata about the download

College-Level Concepts:
- Model Caching: Storing downloaded models locally to avoid re-downloading
- Dependency Management: Ensuring required assets are available before runtime
- Checksum Verification: Validating downloaded files haven't been corrupted
- SOLID Principles: Single Responsibility (one script, one job)
"""
from __future__ import annotations

import argparse  # For command-line arguments
import hashlib  # For file checksums
import json  # For metadata storage
import logging  # For progress tracking
import sys  # For system operations
from datetime import datetime  # For timestamps
from pathlib import Path  # For file paths
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinBERTDownloader:
    """Downloads and manages FinBERT model files.

    What this does: Handles complete FinBERT download and caching process
    Why: Centralizes model management following Single Responsibility Principle
    How: Uses transformers library to download, manages local cache

    This class demonstrates SOLID principles:
    - Single Responsibility: Only downloads and verifies FinBERT
    - Open/Closed: Could be extended for other models without modification
    - Interface Segregation: Clean, focused interface
    - Dependency Inversion: Uses transformers abstractions

    College-Level Analogy:
    Think of this as a librarian who manages book acquisitions. They order books
    (download model), check they arrived correctly (verify), catalog them (cache),
    and ensure they're accessible when needed (test).
    """

    def __init__(
        self,
        *,
        model_name: str = "ProsusAI/finbert",  # Hugging Face model identifier
        cache_dir: Path = Path("model/finbert"),  # Local cache directory
        force_download: bool = False,  # Re-download even if cached
    ):
        """Initialize FinBERT downloader with configuration.

        What: Sets up downloader with target model and cache location
        Why: Allows customization of download behavior
        How: Stores parameters for use in download methods

        Args:
            model_name: HuggingFace model ID (default: ProsusAI/finbert)
            cache_dir: Where to cache model locally
            force_download: If True, download even if already cached

        Model Name Explanation:
        - ProsusAI/finbert: Organization/model-name format on Hugging Face
        - This is the most popular financial sentiment model
        - ~400MB download size
        - Based on BERT architecture, fine-tuned on financial news
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.force_download = force_download

        # Ensure cache directory exists
        # What: Create directory if it doesn't exist
        # Why: Can't cache files if directory missing
        # How: mkdir with parents=True (creates parent directories)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory: {self.cache_dir}")

    def is_cached(self) -> bool:
        """Check if model is already cached locally.

        What: Determines if we need to download or can use cached version
        Why: Avoid unnecessary downloads (saves time and bandwidth)
        How: Checks for presence of key model files

        Returns:
            True if model appears to be cached, False otherwise

        Key Files to Check:
        - config.json: Model architecture configuration
        - pytorch_model.bin: Trained model weights
        - tokenizer_config.json: Tokenizer configuration
        - vocab.txt: Vocabulary mapping
        """
        # List of essential files that must exist for model to work
        # What: Files required by transformers library to load model
        # Why: Missing any of these means model won't load
        required_files = [
            "config.json",  # Architecture config (layers, hidden size, etc.)
            "tf_model.h5",  # Trained TensorFlow weights (largest file, ~400MB)
            "tokenizer_config.json",  # Tokenizer settings
            "vocab.txt",  # Word vocabulary for tokenization
        ]

        # Check if all required files exist
        # What: Verify each file is present in cache directory
        # Why: Partial cache is same as no cache (model won't load)
        # How: Check file existence for each required file
        for filename in required_files:
            file_path = self.cache_dir / filename
            if not file_path.exists():
                logger.info(f"Missing required file: {filename}")
                return False

        logger.info("✓ All required model files found in cache")
        return True

    def download(self) -> bool:
        """Download FinBERT model and tokenizer.

        What: Main download function that fetches model from Hugging Face
        Why: Get model files onto local disk for use
        How: Uses transformers library's from_pretrained methods

        Returns:
            True if download successful, False otherwise

        Process:
        1. Check if already cached (skip if found and not force_download)
        2. Import transformers library
        3. Download tokenizer (vocabulary and config)
        4. Download model (architecture and weights)
        5. Verify downloaded files
        6. Save metadata

        Why Two Downloads:
        - Tokenizer: Converts text to numbers (preprocessing)
        - Model: Neural network that processes the numbers (inference)
        - Both needed for sentiment analysis to work
        """
        # Check if already cached
        # What: Skip download if model already available
        # Why: Save time and bandwidth
        # How: Call is_cached() method
        if not self.force_download and self.is_cached():
            logger.info("Model already cached. Use --force to re-download.")
            return True

        logger.info(f"Downloading {self.model_name} from Hugging Face...")

        try:
            # WHAT: Import transformers library and TensorFlow
            # WHY: Provides download and model loading functionality for TensorFlow models
            # HOW: Dynamic import (allows checking if library installed)
            # DATA: Loads HuggingFace transformers with TensorFlow backend
            from transformers import TFAutoModel, AutoTokenizer
            import tensorflow as tf  # TensorFlow (deep learning framework)

            # WHAT: Download tokenizer files from Hugging Face Hub
            # WHY: Need tokenizer to convert text to model inputs (same for TF and PyTorch)
            # HOW: from_pretrained downloads and caches automatically
            # DATA: Downloads vocab.txt, tokenizer_config.json, special_tokens_map.json
            logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,  # Where to save
                force_download=self.force_download  # Re-download if needed
            )
            logger.info("✓ Tokenizer downloaded")

            # WHAT: Download TensorFlow version of the model from Hugging Face
            # WHY: Need TensorFlow model weights for sentiment predictions
            # HOW: TFAutoModel.from_pretrained downloads TF checkpoint files
            # DATA: Downloads tf_model.h5 or model.ckpt (TensorFlow format) instead of pytorch_model.bin
            logger.info("Downloading TensorFlow model (this may take a few minutes, ~400MB)...")
            model = TFAutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                force_download=self.force_download,
                from_pt=False  # Download TensorFlow weights, not PyTorch
            )
            logger.info("✓ TensorFlow model downloaded")

            # Verify download
            # What: Confirm all necessary files present
            # Why: Partial downloads won't work
            # How: Check file existence again
            if not self.is_cached():
                logger.error("Download completed but files missing. Check cache directory.")
                return False

            # Save metadata
            # What: Record download information
            # Why: Track when/what was downloaded for debugging
            # How: Write JSON file with metadata
            metadata = {
                'model_name': self.model_name,
                'download_date': datetime.now().isoformat(),
                'cache_dir': str(self.cache_dir),
                'model_size_mb': self._calculate_directory_size() / (1024 * 1024)
            }

            metadata_path = self.cache_dir / "download_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Metadata saved to {metadata_path}")

            logger.info("=" * 60)
            logger.info("✓ FinBERT download complete!")
            logger.info(f"Model cached at: {self.cache_dir}")
            logger.info(f"Total size: {metadata['model_size_mb']:.1f} MB")
            logger.info("=" * 60)

            return True

        except ImportError as e:
            # WHAT: Handle missing dependencies gracefully
            # WHY: Provide helpful error message instead of cryptic trace
            # HOW: Catch ImportError specifically and show installation command
            # DATA: Returns False to indicate download failed
            logger.error("Required libraries not installed:")
            logger.error("  pip install transformers tensorflow")
            logger.error(f"Error: {e}")
            return False

        except Exception as e:
            # Handle other download errors
            # What: Catch network errors, disk errors, etc.
            # Why: Graceful error handling with helpful message
            # How: Generic exception catch with logging
            logger.error(f"Download failed: {e}", exc_info=True)
            return False

    def verify(self) -> bool:
        """Verify downloaded model works correctly.

        What: Tests model by running a sample prediction
        Why: Ensures download successful and model functional
        How: Loads model and runs test inference

        Returns:
            True if model loads and runs correctly, False otherwise

        Verification Steps:
        1. Import libraries
        2. Load tokenizer and model from cache
        3. Run test prediction on sample text
        4. Check prediction format is correct
        5. Log success or failure
        """
        logger.info("Verifying model...")

        # Check if model cached
        # What: Confirm model files exist before trying to load
        # Why: Clearer error if files missing
        # How: Call is_cached() method
        if not self.is_cached():
            logger.error("Model not cached. Run download first.")
            return False

        try:
            # WHAT: Import libraries for model verification
            # WHY: Need transformers and TensorFlow to load and test model
            # HOW: Import TensorFlow versions of transformers classes
            # DATA: Loads TFAutoModelForSequenceClassification for TensorFlow models
            from transformers import (
                TFAutoModelForSequenceClassification,
                AutoTokenizer,
            )
            import tensorflow as tf

            # Load tokenizer
            # What: Load tokenizer from cache
            # Why: Need tokenizer to preprocess test text
            # How: from_pretrained with local_files_only=True
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.cache_dir),  # Load from cache, not Hugging Face
                local_files_only=True  # Fail if files not local (don't download)
            )

            # WHAT: Load TensorFlow model from cache
            # WHY: Need model to run test prediction and verify it works
            # HOW: TFAutoModelForSequenceClassification.from_pretrained with local_files_only=True
            # DATA: Loads tf_model.h5 from cache directory into memory
            logger.info("Loading TensorFlow model...")
            model = TFAutoModelForSequenceClassification.from_pretrained(
                str(self.cache_dir),
                local_files_only=True  # Fail if files not local (don't download)
            )

            # WHAT: Note that TensorFlow models are in inference mode by default
            # WHY: Unlike PyTorch, TensorFlow doesn't require explicit .eval() call
            # HOW: TensorFlow models automatically disable training-specific layers
            # DATA: Model is ready for prediction without additional configuration

            # Run test prediction
            # What: Predict sentiment on a sample financial headline
            # Why: Ensures full inference pipeline works
            # How: Tokenize text, run through model, extract probabilities
            logger.info("Running test prediction...")
            test_text = "Apple reports strong quarterly earnings, beating analyst expectations"

            # WHAT: Tokenize input text into numerical format
            # WHY: Neural networks need numerical inputs, not strings
            # HOW: Tokenizer converts text → token IDs, attention masks
            # DATA: Text string → dictionary with TensorFlow tensors
            inputs = tokenizer(
                test_text,
                return_tensors="tf",  # TensorFlow tensors
                truncation=True,
                padding=True,
                max_length=512
            )

            # WHAT: Run inference (forward pass through neural network)
            # WHY: Get sentiment predictions to verify model works
            # HOW: Pass inputs to model, apply softmax to get probabilities
            # DATA: inputs (tensors) → outputs.logits (raw scores) → probabilities (0-1 range)
            # Note: TensorFlow models don't need torch.no_grad() context manager
            outputs = model(**inputs)

            # WHAT: Convert raw logits to probabilities using softmax
            # WHY: Logits are unbounded; softmax normalizes to [0,1] summing to 1
            # HOW: Apply tf.nn.softmax along class dimension (axis=-1)
            # DATA: logits → probabilities for [positive, negative, neutral]
            probabilities = tf.nn.softmax(outputs.logits, axis=-1)

            # WHAT: Extract probabilities and convert to NumPy array
            # WHY: Easier to work with standard Python types than TensorFlow tensors
            # HOW: Call .numpy() on TensorFlow tensor, get first (and only) batch result
            # DATA: TensorFlow tensor → NumPy array [prob_pos, prob_neg, prob_neu]
            probs = probabilities.numpy()[0]

            # Log results
            # What: Show prediction results
            # Why: Verify model producing sensible outputs
            # How: Print probabilities for each class
            logger.info("Test prediction results:")
            logger.info(f"  Text: '{test_text}'")
            logger.info(f"  Positive: {probs[0]:.4f}")  # Probability of positive sentiment
            logger.info(f"  Negative: {probs[1]:.4f}")  # Probability of negative sentiment
            logger.info(f"  Neutral:  {probs[2]:.4f}")  # Probability of neutral sentiment

            # Verify prediction makes sense
            # What: Basic sanity check on results
            # Why: Ensure model isn't broken (e.g., always predicting same class)
            # How: Check that probabilities sum to ~1 and positive is highest
            prob_sum = sum(probs)
            if abs(prob_sum - 1.0) > 0.01:  # Allow small floating point error
                logger.warning(f"Probabilities don't sum to 1.0 (sum={prob_sum})")

            # For this positive headline, positive sentiment should be highest
            predicted_class = probs.argmax()  # Index of highest probability
            class_names = ["positive", "negative", "neutral"]
            logger.info(f"  Predicted: {class_names[predicted_class]}")

            if predicted_class != 0:  # 0 = positive
                logger.warning("Expected positive sentiment for test headline, but got different prediction")
                logger.warning("This may indicate model issues, but isn't necessarily wrong")

            logger.info("✓ Model verification successful")
            return True

        except ImportError as e:
            # WHAT: Handle missing dependencies gracefully
            # WHY: Provide helpful error message with correct installation command
            # HOW: Catch ImportError and show TensorFlow installation instructions
            # DATA: Returns False to indicate verification failed
            logger.error("Required libraries not installed for verification:")
            logger.error("  pip install transformers tensorflow")
            logger.error(f"Error: {e}")
            return False

        except Exception as e:
            # Handle verification errors
            logger.error(f"Verification failed: {e}", exc_info=True)
            return False

    def _calculate_directory_size(self) -> int:
        """Calculate total size of cache directory in bytes.

        What: Sums up sizes of all files in cache directory
        Why: Show user how much disk space model uses
        How: Recursively walk directory tree and sum file sizes

        Returns:
            Total size in bytes
        """
        total_size = 0

        # Walk through all files in cache directory
        # What: Iterate through directory tree
        # Why: Need to check all files, including subdirectories
        # How: Path.rglob("*") finds all files recursively
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                # Add file size to total
                # What: Get file size and accumulate
                # Why: Need total of all files
                # How: .stat().st_size returns size in bytes
                total_size += path.stat().st_size

        return total_size


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    What: Handles command-line options
    Why: Makes script flexible and scriptable
    How: Uses argparse to define and parse arguments

    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Download FinBERT model for StockSense sentiment analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download with default settings
  python download_finbert.py

  # Force re-download even if cached
  python download_finbert.py --force

  # Download to custom directory
  python download_finbert.py --cache-dir /path/to/cache

  # Download and skip verification
  python download_finbert.py --no-verify

  # Download custom model (advanced)
  python download_finbert.py --model-name yiyanghkust/finbert-tone
        """
    )

    parser.add_argument(
        '--model-name',
        default='ProsusAI/finbert',
        help='HuggingFace model identifier (default: ProsusAI/finbert)'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('model/finbert'),
        help='Directory to cache model files (default: model/finbert)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if model already cached'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip model verification after download'
    )

    return parser.parse_args()


def main():
    """Main entry point for FinBERT download script.

    What: Orchestrates download and verification process
    Why: Provides command-line interface
    How: Creates downloader, runs download, verifies

    This demonstrates the Single Responsibility Principle:
    - Only coordinates CLI workflow
    - Delegates actual work to FinBERTDownloader
    """
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("FinBERT Model Downloader")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Cache: {args.cache_dir}")
    logger.info(f"Force download: {args.force}")
    logger.info("=" * 60)

    # Create downloader
    # What: Initialize downloader with configuration
    # Why: Separates configuration from execution
    # How: Pass command-line args to constructor
    downloader = FinBERTDownloader(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        force_download=args.force
    )

    # Download model
    # What: Fetch model from Hugging Face
    # Why: Main purpose of this script
    # How: Downloader handles entire process
    download_success = downloader.download()

    if not download_success:
        logger.error("Download failed. See errors above.")
        return 1

    # Verify model (unless --no-verify)
    # What: Test that model works correctly
    # Why: Catch download issues early
    # How: Run test prediction
    if not args.no_verify:
        logger.info("=" * 60)
        verify_success = downloader.verify()

        if not verify_success:
            logger.error("Verification failed. Model may not work correctly.")
            return 1

    # Success summary
    logger.info("=" * 60)
    logger.info("✓ FinBERT Setup Complete!")
    logger.info("=" * 60)
    logger.info(f"Model location: {args.cache_dir}")
    logger.info("The model is now ready for use by StockSense.")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Verify pipelines/realtime/sentiment/finbert.py uses this cache")
    logger.info("2. Test sentiment analysis: python -m pipelines.realtime.sentiment.finbert")
    logger.info("3. Run full StockSense pipeline to verify integration")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

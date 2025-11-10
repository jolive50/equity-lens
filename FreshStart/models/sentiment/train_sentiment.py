"""Fine-tuning script for sentiment models (FinBERT and RoBERTa).

Usage:
    python train_sentiment.py --model finbert --data training_data.csv --output ./finbert_finetuned
    python train_sentiment.py --model roberta --data training_data.csv --output ./roberta_finetuned

CSV format:
    text,label
    "Apple beats earnings expectations",0
    "Market crashes due to uncertainty",1
    "Company maintains guidance",2

Labels: 0=positive, 1=negative, 2=neutral (FinBERT)
        0=negative, 1=neutral, 2=positive (RoBERTa)
"""

import argparse
import pandas as pd
from typing import List, Tuple
from .finbert_model import FinBERTModel
from .roberta_model import RoBERTaModel


def load_training_data(csv_path: str) -> Tuple[List[str], List[int]]:
    """Load training data from CSV file.

    Args:
        csv_path: Path to CSV file with columns: text, label

    Returns:
        Tuple of (texts, labels)

    Raises:
        ValueError: If CSV format is invalid
    """
    try:
        df = pd.read_csv(csv_path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have 'text' and 'label' columns")

        texts = df["text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()

        return texts, labels

    except Exception as e:
        raise ValueError(f"Failed to load training data: {e}") from e


def split_data(
    texts: List[str],
    labels: List[int],
    val_split: float = 0.2
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Split data into training and validation sets.

    Args:
        texts: All texts
        labels: All labels
        val_split: Fraction to use for validation (default 0.2)

    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels)
    """
    n = len(texts)
    split_idx = int(n * (1 - val_split))

    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    val_texts = texts[split_idx:]
    val_labels = labels[split_idx:]

    return train_texts, train_labels, val_texts, val_labels


def fine_tune_finbert(
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_split: float = 0.2
) -> None:
    """Fine-tune FinBERT model on custom dataset.

    Args:
        data_path: Path to training CSV
        output_dir: Directory to save fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        val_split: Validation split fraction
    """
    print("Loading FinBERT model...")
    model = FinBERTModel()

    print(f"Loading training data from {data_path}...")
    texts, labels = load_training_data(data_path)

    print(f"Loaded {len(texts)} examples")
    print(f"Splitting into train/validation (val_split={val_split})...")
    train_texts, train_labels, val_texts, val_labels = split_data(
        texts, labels, val_split
    )

    print(f"Training set: {len(train_texts)} examples")
    print(f"Validation set: {len(val_texts)} examples")

    print("\nStarting fine-tuning...")
    history = model.fine_tune(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir
    )

    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print("=" * 60)
    print(f"Epochs: {history['epochs']}")
    print(f"Final training loss: {history['final_train_loss']:.4f}")
    print(f"Final training accuracy: {history['final_train_accuracy']:.4f}")
    if history['final_val_loss']:
        print(f"Final validation loss: {history['final_val_loss']:.4f}")
        print(f"Final validation accuracy: {history['final_val_accuracy']:.4f}")
    print(f"\nWeights saved to: {history['weights_saved_to']}")
    print("=" * 60)


def fine_tune_roberta(
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_split: float = 0.2
) -> None:
    """Fine-tune RoBERTa model on custom dataset.

    Args:
        data_path: Path to training CSV
        output_dir: Directory to save fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        val_split: Validation split fraction
    """
    print("Loading RoBERTa model...")
    model = RoBERTaModel()

    print(f"Loading training data from {data_path}...")
    texts, labels = load_training_data(data_path)

    print(f"Loaded {len(texts)} examples")
    print(f"Splitting into train/validation (val_split={val_split})...")
    train_texts, train_labels, val_texts, val_labels = split_data(
        texts, labels, val_split
    )

    print(f"Training set: {len(train_texts)} examples")
    print(f"Validation set: {len(val_texts)} examples")

    print("\nStarting fine-tuning...")
    history = model.fine_tune(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir
    )

    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print("=" * 60)
    print(f"Epochs: {history['epochs']}")
    print(f"Final training loss: {history['final_train_loss']:.4f}")
    print(f"Final training accuracy: {history['final_train_accuracy']:.4f}")
    if history['final_val_loss']:
        print(f"Final validation loss: {history['final_val_loss']:.4f}")
        print(f"Final validation accuracy: {history['final_val_accuracy']:.4f}")
    print(f"\nWeights saved to: {history['weights_saved_to']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune sentiment models (FinBERT or RoBERTa)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["finbert", "roberta"],
        help="Model to fine-tune (finbert or roberta)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training CSV (must have 'text' and 'label' columns)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for fine-tuned model weights"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)"
    )

    args = parser.parse_args()

    if args.model == "finbert":
        fine_tune_finbert(
            data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split
        )
    elif args.model == "roberta":
        fine_tune_roberta(
            data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()

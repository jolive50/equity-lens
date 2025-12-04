"""
Quick checker to verify sentiment model files and cached Hugging Face weights.

Usage:
    python -m utils.check_sentiment_models
"""
from pathlib import Path
import os


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sentiment_dir = root / "models" / "sentiment"
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

    print(f"Sentiment model files in {sentiment_dir}:")
    for name in ["finbert_model.py", "roberta_model.py", "deberta_model.py"]:
        path = sentiment_dir / name
        print(f"  {name}: {'present' if path.exists() else 'missing'}")

    print(f"\nHugging Face cache at {hf_cache}: {'present' if hf_cache.exists() else 'missing'}")
    if hf_cache.exists():
        for repo in sorted(hf_cache.glob("models--*")):
            print(f"  cache: {repo.name}")

    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    print("\nHUGGINGFACE_HUB_TOKEN set:", bool(token))


if __name__ == "__main__":
    main()

import pytest
import os
import sys
import json
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.database import Database


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name

    db = Database(db_path)
    yield db

    # Close database connections before cleanup
    db.close()

    # Small delay to ensure Windows releases file lock
    import time
    time.sleep(0.1)

    # Clean up temporary database file
    try:
        if os.path.exists(db_path):
            os.unlink(db_path)
    except PermissionError:
        # If still locked, try to mark for deletion on next reboot
        pass


@pytest.fixture
def sample_price_data():
    """Load sample price data from fixtures."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_prices.json"
    if fixtures_path.exists():
        with open(fixtures_path, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
    else:
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        return pd.DataFrame({
            'Open': [100 + i for i in range(30)],
            'High': [105 + i for i in range(30)],
            'Low': [95 + i for i in range(30)],
            'Close': [102 + i for i in range(30)],
            'Volume': [1000000 + i*10000 for i in range(30)]
        }, index=dates)


@pytest.fixture
def sample_news_data():
    """Load sample news data from fixtures."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_news.json"
    if fixtures_path.exists():
        with open(fixtures_path, 'r') as f:
            return json.load(f)['articles']
    else:
        return [
            {
                'title': 'Company reports strong earnings',
                'url': 'https://example.com/article1',
                'published_at': datetime.now().isoformat(),
                'source': 'Financial News',
                'summary': 'The company exceeded expectations with record profits.',
                'sentiment_label': 'positive',
                'sentiment_score': 0.85
            },
            {
                'title': 'Market analysts predict growth',
                'url': 'https://example.com/article2',
                'published_at': (datetime.now() - timedelta(days=1)).isoformat(),
                'source': 'Market Watch',
                'summary': 'Analysts are optimistic about future performance.',
                'sentiment_label': 'positive',
                'sentiment_score': 0.72
            }
        ]


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing."""
    return {
        'prediction_direction': 'up',
        'prediction_confidence': 0.75,
        'prediction_model': 'LSTM',
        'sentiment_score': 0.65,
        'sentiment_label': 'positive',
        'sentiment_model': 'FinBERT',
        'explanation': 'Strong technical indicators and positive sentiment',
        'reflection_warnings': None,
        'raw_data': {'test': 'data'}
    }


@pytest.fixture
def mock_ticker():
    """Default test ticker symbol."""
    return "AAPL"


@pytest.fixture
def expected_outputs():
    """Load expected test outputs from fixtures."""
    fixtures_path = Path(__file__).parent / "fixtures" / "expected_outputs.json"
    if fixtures_path.exists():
        with open(fixtures_path, 'r') as f:
            return json.load(f)
    return {}

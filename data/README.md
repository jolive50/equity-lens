# Data Directory

**⚠️ IMPORTANT: Data files are NOT in git!**

This directory contains large datasets that are **excluded from version control** to keep the repository lightweight.

---

## Why Data is Excluded

- **Size**: Training data is ~100+ MB (too large for git)
- **Reproducibility**: Anyone can download the same data using our scripts
- **Flexibility**: Each team member can choose which datasets to download
- **Best Practice**: Never commit large data files to git

---

## How to Download Data

### Quick Start (Recommended)

```bash
# 1. Set up Kaggle credentials (one-time setup)
#    Get your kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2. Download S&P 500 price data (~56 MB)
python scripts/download_kaggle_data.py

# 3. Prepare specific stocks for training
python scripts/prepare_kaggle_data.py --tickers AAPL MSFT GOOGL AMZN NVDA

# 4. (Optional) Download fundamental data (~40 MB)
python scripts/download_fundamentals.py --dataset sp500
python scripts/download_fundamentals.py --dataset sec
```

### Detailed Instructions

See **[SETUP_GUIDE.md](../SETUP_GUIDE.md)** in the root directory for:
- Complete setup instructions
- How to get Kaggle API credentials
- Alternative data sources (yfinance, etc.)
- Troubleshooting tips

---

## Directory Structure

After downloading data, you'll have:

```
data/
├── training/
│   ├── raw/                        # Raw downloaded data
│   │   └── kaggle_sp500/           # S&P 500 data from Kaggle
│   │       └── all_stocks_5yr.csv  # ~56 MB consolidated file
│   ├── prepared/                   # Individual ticker CSVs
│   │   ├── AAPL.csv
│   │   ├── MSFT.csv
│   │   └── ...
│   ├── features/                   # Engineered features (after build_training_dataset.py)
│   │   └── features.csv
│   ├── labels/                     # Training labels (after build_training_dataset.py)
│   │   └── labels.csv
│   └── ticker_list.txt             # List of prepared tickers
└── fundamentals/                   # Fundamental financial data
    ├── sp500_financials/
    │   └── financials.csv          # 505 companies, 14 metrics
    └── sec_indicators/             # 225 metrics from SEC filings
        ├── 2014_Financial_Data.csv
        ├── 2015_Financial_Data.csv
        ├── 2016_Financial_Data.csv
        ├── 2017_Financial_Data.csv
        └── 2018_Financial_Data.csv
```

---

## What Data Do You Need?

### For Basic Training/Testing
```bash
# Just price data (minimum required)
python scripts/download_kaggle_data.py
python scripts/prepare_kaggle_data.py --tickers AAPL MSFT GOOGL
```
**Size**: ~10-20 MB

### For Full Model Training
```bash
# Price data + fundamentals
python scripts/download_kaggle_data.py
python scripts/prepare_kaggle_data.py --top 50
python scripts/download_fundamentals.py --dataset sp500
```
**Size**: ~100 MB

### For Complete Dataset (All Features)
```bash
# Everything
python scripts/download_kaggle_data.py
python scripts/prepare_kaggle_data.py --all
python scripts/download_fundamentals.py --all
```
**Size**: ~150 MB

---

## Data Inventory

See **[DATA_INVENTORY.md](../DATA_INVENTORY.md)** for complete list of:
- 225 fundamental metrics available
- Which metrics match course requirements
- How to access specific data points

---

## Troubleshooting

### "No Kaggle credentials found"
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Move to `~/.kaggle/kaggle.json`
5. Run: `chmod 600 ~/.kaggle/kaggle.json`

### "Dataset not found"
- Check internet connection
- Verify Kaggle credentials are correct
- Dataset may have been renamed/removed - try alternative datasets

### "Out of disk space"
- Full dataset is ~150 MB
- Download only what you need using `--tickers` flag
- Clean up old data: `rm -rf data/training/raw/`

---

## Need Help?

1. Read [SETUP_GUIDE.md](../SETUP_GUIDE.md)
2. Check [DATA_INVENTORY.md](../DATA_INVENTORY.md)
3. Ask your team members
4. Check Kaggle dataset documentation

---

## Important Notes

- ✅ **Data is reproducible** - everyone gets the same datasets
- ✅ **Data is excluded from git** - won't bloat repository
- ✅ **Data is free** - no paid API keys required
- ⚠️ **Don't commit data files** - already in .gitignore
- ⚠️ **Don't commit credentials** - kaggle.json stays local

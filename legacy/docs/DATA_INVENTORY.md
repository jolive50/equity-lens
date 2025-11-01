# StockSense Data Inventory

Complete inventory of all downloaded data and available metrics.

---

## 📊 Price Data (Historical OHLCV)

### Source: Kaggle S&P 500 Dataset
- **Location**: `data/training/raw/kaggle_sp500/`
- **Files**: 510 CSV files (one consolidated + 509 individual tickers)
- **Size**: 56.44 MB
- **Date Range**: 2013-02-08 to 2018-02-07 (~5 years)
- **Frequency**: Daily
- **Tickers**: 505 S&P 500 companies

### Columns Available:
```
- date          # Trading date
- open          # Opening price
- high          # Highest price
- low           # Lowest price
- close         # Closing price
- volume        # Trading volume
- Name          # Ticker symbol
```

### Prepared Tickers (Ready for Training):
**Location**: `data/training/prepared/`

Currently prepared: **8 tech stocks**
```
AAPL    - Apple Inc.
AMD     - Advanced Micro Devices
AMZN    - Amazon.com Inc.
GOOGL   - Alphabet Inc.
INTC    - Intel Corporation
MSFT    - Microsoft Corporation
NFLX    - Netflix Inc.
NVDA    - NVIDIA Corporation
```

Each has **~1,259 rows** of daily OHLCV data.

---

## 💰 Fundamental Data (Financial Metrics)

### 1. S&P 500 Financial Information

**Location**: `data/fundamentals/sp500_financials/financials.csv`
- **Companies**: 505 S&P 500 stocks
- **Metrics**: 14 key financial indicators

#### Available Metrics:
```
✓ Symbol                # Ticker
✓ Name                  # Company name
✓ Sector                # Business sector
✓ Price                 # Current stock price
✓ Price/Earnings        # P/E Ratio
✓ Dividend Yield        # Annual dividend %
✓ Earnings/Share        # EPS
✓ 52 Week Low           # Yearly low price
✓ 52 Week High          # Yearly high price
✓ Market Cap            # Total market value
✓ EBITDA                # Earnings before interest, tax, depreciation, amortization
✓ Price/Sales           # P/S Ratio
✓ Price/Book            # P/B Ratio
✓ SEC Filings           # Link to SEC documents
```

---

### 2. SEC Financial Indicators (200+ Metrics)

**Location**: `data/fundamentals/sec_indicators/`
- **Files**: 5 yearly datasets (2014-2018)
- **Size**: 39 MB total
- **Companies**: 4,392 US stocks per year
- **Metrics**: **225 financial indicators!**

#### Complete Metrics List:

**Income Statement (Profitability)**
```
✓ Revenue                           # Total sales
✓ Revenue Growth                    # YoY revenue growth %
✓ Cost of Revenue                   # COGS (cost of goods sold)
✓ Gross Profit                      # Revenue - COGS
✓ R&D Expenses                      # Research & Development spending
✓ SG&A Expense                      # Selling, General & Administrative
✓ Operating Expenses                # Total operating costs
✓ Operating Income                  # Profit from operations
✓ Interest Expense                  # Debt interest payments
✓ Earnings before Tax (EBT)         # Income before taxes
✓ Income Tax Expense                # Taxes paid
✓ Net Income                        # Bottom line profit
✓ EPS (Earnings Per Share)          # Profit per share
✓ EPS Diluted                       # EPS accounting for stock options
✓ Dividend per Share                # Dividend payout per share
```

**Margins (Profitability Ratios)**
```
✓ Gross Margin                      # Gross profit / Revenue
✓ EBITDA Margin                     # EBITDA / Revenue
✓ EBIT Margin                       # Operating income / Revenue
✓ Profit Margin                     # Net income / Revenue
✓ Free Cash Flow Margin             # FCF / Revenue
✓ Net Profit Margin                 # Same as profit margin
✓ Earnings Before Tax Margin        # EBT / Revenue
```

**Balance Sheet (Assets & Liabilities)**
```
✓ Cash and cash equivalents         # Liquid cash
✓ Short-term investments            # Marketable securities
✓ Cash and short-term investments   # Total liquid assets
✓ Receivables                       # Money owed to company
✓ Inventories                       # Products in stock
✓ Total current assets              # Assets convertible to cash <1yr
✓ Property, Plant & Equipment (PP&E)# Physical assets
✓ Goodwill and Intangible Assets    # Non-physical assets
✓ Long-term investments             # Strategic holdings
✓ Tax assets                        # Tax credits/refunds
✓ Total non-current assets          # Long-term assets
✓ Total assets                      # Everything company owns
✓ Payables                          # Money company owes
✓ Short-term debt                   # Debt due <1yr
✓ Total current liabilities         # Obligations due <1yr
✓ Long-term debt                    # Debt due >1yr
✓ Total debt                        # All debt
✓ Deferred revenue                  # Prepayments from customers
✓ Tax Liabilities                   # Taxes owed
✓ Deposit Liabilities               # Customer deposits
✓ Total non-current liabilities     # Long-term obligations
✓ Total liabilities                 # Everything company owes
✓ Retained earnings                 # Cumulative profits kept
✓ Total shareholders equity         # Net worth (assets - liabilities)
✓ Accumulated other comprehensive income
```

**Cash Flow Statement**
```
✓ Operating Cash Flow               # Cash from business operations
✓ Capital Expenditure               # Money spent on PP&E
✓ Free Cash Flow                    # Operating CF - CapEx
✓ Investing Cash Flow               # Cash from investments
✓ Financing Cash Flow               # Cash from debt/equity
✓ Net change in cash                # Total cash change
✓ Depreciation & Amortization       # Non-cash expenses
✓ Stock-based compensation          # Employee stock grants
```

**Valuation Ratios**
```
✓ P/E Ratio (Price/Earnings)        # Market value per $1 earnings
✓ P/S Ratio (Price/Sales)           # Market value per $1 revenue
✓ P/B Ratio (Price/Book)            # Market value / Book value
✓ EV/Sales                          # Enterprise value / Revenue
✓ EV/EBITDA                         # Enterprise value / EBITDA
✓ PEG Ratio                         # P/E / Growth rate
✓ Enterprise Value (EV)             # Market cap + debt - cash
✓ Market Capitalization             # Share price × shares outstanding
```

**Efficiency Ratios**
```
✓ Asset Turnover                    # Revenue / Total assets
✓ Inventory Turnover                # COGS / Average inventory
✓ Receivables Turnover              # Revenue / Receivables
✓ Payables Turnover                 # COGS / Payables
✓ Days Sales Outstanding            # Days to collect payment
✓ Days Inventory Outstanding        # Days inventory held
✓ Days Payables Outstanding         # Days to pay suppliers
✓ Cash Conversion Cycle             # Working capital efficiency
```

**Profitability Ratios**
```
✓ Return on Assets (ROA)            # Net income / Total assets
✓ Return on Equity (ROE)            # Net income / Shareholders equity
✓ Return on Invested Capital (ROIC) # Operating income / Invested capital
✓ Return on Tangible Assets         # Income / Tangible assets
✓ Gross Profit Margin               # Gross profit / Revenue
✓ Operating Margin                  # Operating income / Revenue
✓ Pretax Profit Margin              # EBT / Revenue
```

**Leverage Ratios**
```
✓ Debt/Equity Ratio                 # Total debt / Shareholders equity
✓ Debt/Assets Ratio                 # Total debt / Total assets
✓ Debt/EBITDA                       # Total debt / EBITDA
✓ Interest Coverage                 # EBIT / Interest expense
✓ Current Ratio                     # Current assets / Current liabilities
✓ Quick Ratio                       # (Current assets - Inventory) / Current liabilities
✓ Cash Ratio                        # Cash / Current liabilities
```

**Growth Metrics**
```
✓ Revenue Growth (YoY)              # Year-over-year revenue change %
✓ Earnings Growth                   # YoY net income change %
✓ EPS Growth                        # YoY EPS change %
✓ Operating Income Growth           # YoY operating income change %
✓ Free Cash Flow Growth             # YoY FCF change %
✓ Asset Growth                      # YoY total assets change %
✓ Book Value Growth                 # YoY equity change %
```

**Per Share Metrics**
```
✓ Earnings Per Share (EPS)          # Net income / Shares
✓ Book Value Per Share              # Equity / Shares
✓ Sales Per Share                   # Revenue / Shares
✓ Cash Flow Per Share               # Operating CF / Shares
✓ Free Cash Flow Per Share          # FCF / Shares
✓ Tangible Book Value Per Share     # (Equity - Intangibles) / Shares
```

**Additional Metrics** (175-225)
```
✓ Working Capital                   # Current assets - Current liabilities
✓ Tangible Asset Value              # Total assets - Intangibles
✓ Net Debt                          # Total debt - Cash
✓ Invested Capital                  # Debt + Equity
✓ Average Assets                    # (Beginning + Ending assets) / 2
✓ Average Equity                    # (Beginning + Ending equity) / 2
✓ Income Quality                    # Operating CF / Net income
✓ Dividend Payout Ratio             # Dividends / Net income
✓ Retention Ratio                   # (1 - Payout ratio)
✓ CAPEX/Revenue                     # Capital intensity
✓ CAPEX/Operating CF                # Reinvestment rate
✓ Effective Tax Rate                # Tax expense / EBT
✓ Employee Count                    # Number of employees
✓ Revenue Per Employee              # Productivity metric
✓ ... and 150+ more!
```

---

## 🤖 ML Models

### FinBERT (Sentiment Analysis)
- **Location**: Hugging Face cache (`~/.cache/huggingface/` or `model/finbert/`)
- **Size**: ~440 MB
- **Purpose**: Analyze financial news sentiment
- **Output**: Positive/Negative/Neutral probabilities

### Forecaster (Stock Direction Prediction)
- **Location**: `model/artifacts/` (after training)
- **Model Type**: Gradient Boosting Classifier
- **Input Features**: 15 technical indicators + optional fundamentals
- **Output**: Up/Down/Neutral probabilities with confidence

---

## 📁 Complete Directory Structure

```
capstone/
├── data/
│   ├── training/
│   │   ├── raw/
│   │   │   └── kaggle_sp500/
│   │   │       ├── all_stocks_5yr.csv      # 619,040 rows
│   │   │       └── [509 individual CSVs]    # One per ticker
│   │   ├── prepared/
│   │   │   ├── AAPL.csv                     # 1,259 rows
│   │   │   ├── AMD.csv
│   │   │   ├── AMZN.csv
│   │   │   ├── GOOGL.csv
│   │   │   ├── INTC.csv
│   │   │   ├── MSFT.csv
│   │   │   ├── NFLX.csv
│   │   │   └── NVDA.csv
│   │   └── ticker_list.txt                  # 8 tickers
│   └── fundamentals/
│       ├── sp500_financials/
│       │   └── financials.csv               # 505 companies, 14 metrics
│       └── sec_indicators/
│           ├── 2014_Financial_Data.csv      # 225 metrics
│           ├── 2015_Financial_Data.csv
│           ├── 2016_Financial_Data.csv
│           ├── 2017_Financial_Data.csv
│           └── 2018_Financial_Data.csv
└── model/
    ├── finbert/                             # FinBERT cache (~440 MB)
    └── artifacts/                           # Trained forecaster (after training)
```

---

## 📊 Data Quality Summary

### Price Data Coverage:
- **Time Period**: 5 years (2013-2018)
- **Granularity**: Daily
- **Missing Data**: Minimal (cleaned dataset)
- **Tickers**: 505 S&P 500 companies

### Fundamental Data Coverage:
- **S&P 500 Financials**: Current/recent metrics for 505 companies
- **SEC Indicators**: Historical metrics (2014-2018) for 4,392+ companies
- **Total Unique Metrics**: 225+ financial indicators
- **Data Quality**: High (sourced from SEC filings)

### Overlap:
- Price + S&P 500 Fundamentals: **505 tickers** overlap
- Price + SEC Indicators: **~400-500 tickers** overlap (varies by year)

---

## 🎯 What You Can Do With This Data

### 1. Basic Price Prediction
Use OHLCV data + technical indicators:
```bash
python scripts/build_training_dataset.py --tickers AAPL MSFT GOOGL
python scripts/train_forecaster.py
```

### 2. Enhanced Prediction with Fundamentals
Combine price data + fundamental metrics:
```bash
# Merge fundamentals with prices
python scripts/download_fundamentals.py --dataset sp500 --merge-with-prices data/training/prepared

# Train with enhanced features
python scripts/build_training_dataset.py --include-fundamentals
python scripts/train_forecaster.py
```

### 3. Comprehensive Analysis
Use all available data:
- **225 fundamental metrics** (valuation, profitability, growth)
- **15 technical indicators** (RSI, MACD, Bollinger Bands, etc.)
- **Sentiment analysis** (FinBERT on news)
- **Smart money tracking** (institutional holdings, insider trades)

---

## 🔧 Next Steps

1. ✅ **Downloaded**: Price data (510 CSV files, 56 MB)
2. ✅ **Downloaded**: S&P 500 fundamentals (505 companies, 14 metrics)
3. ✅ **Downloaded**: SEC indicators (225 metrics, 2014-2018)
4. ✅ **Downloaded**: FinBERT model (440 MB)
5. ⏭️ **TODO**: Merge fundamentals with price data
6. ⏭️ **TODO**: Update feature engineering to include fundamentals
7. ⏭️ **TODO**: Train enhanced model with all features
8. ⏭️ **TODO**: Evaluate impact of fundamentals on predictions

---

## 📝 Notes

- All downloaded data is **excluded from git** (see `.gitignore`)
- Data can be **re-downloaded** anytime using scripts
- **Credentials are secure** - never committed to repo
- **Free data sources** - no paid API keys required for training

---

## 🆘 Support

See `SETUP_GUIDE.md` for detailed setup instructions.

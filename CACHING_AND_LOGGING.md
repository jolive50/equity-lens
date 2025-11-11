# Caching and Logging Implementation

## Summary of Changes

This document describes the comprehensive caching and logging system implemented for FreshStart.

## Problems Fixed

### 1. **No Persistent Logs**
- **Before:** All logs only went to console (lost on restart)
- **After:** Logs saved to `logs/freshstart.log` with 10MB rotation (keeps 5 backups)

### 2. **Caching Infrastructure Not Used**
- **Before:** Database cache methods existed but were never called - every request fetched fresh data
- **After:** Workflow checks cache first, only fetches if cache is stale (>1 hour old)

### 3. **No Database Logging**
- **Before:** `storage/database.py` had zero logging statements
- **After:** Comprehensive logging for all cache operations with hit/miss tracking

### 4. **No Cache Performance Visibility**
- **Before:** No way to know if caching was working
- **After:** Detailed cache metrics with hit rates and performance tracking

---

## New Components

### 1. Centralized Logging Configuration
**File:** `utils/logging_config.py`

Features:
- File-based logging with rotation (10MB max, 5 backups)
- Console + file output
- Structured format: `YYYY-MM-DD HH:MM:SS - module - LEVEL - message`
- Automatic `logs/` directory creation

Usage:
```python
from utils.logging_config import setup_logging, get_logger

# One-time setup (in main.py)
setup_logging(name="freshstart", log_dir="logs", log_level=logging.INFO)

# Get logger in any module
logger = get_logger(__name__)
logger.info("Your message here")
```

### 2. Database Cache Enhancements
**File:** `storage/database.py`

New methods:
- `is_price_cache_fresh(ticker, max_age_hours=1)` - Check if cached prices are fresh
- `is_news_cache_fresh(ticker, max_age_hours=1)` - Check if cached news is fresh

Enhanced methods with logging:
- `cache_prices()` - Now logs: "Cached 66 price rows for AAPL (0 errors)"
- `get_cached_prices()` - Now logs: "Cache HIT for AAPL prices (66 rows found)"
- `cache_news()` - Now logs: "Cached 50 news articles for AAPL (0 errors)"
- `get_cached_news()` - Now logs: "Cache HIT for AAPL news (50 articles found)"

### 3. Workflow Caching Integration
**File:** `coordinator/workflow.py`

The `fetch_data()` function now:

**For Price Data:**
1. Check if cache is fresh (`is_price_cache_fresh()`)
2. If fresh → use cached data
3. If stale/missing → fetch from API and cache it
4. Log: "Using cached price data" or "Fetched X days of data"

**For News Data:**
1. Check if cache is fresh (`is_news_cache_fresh()`)
2. If fresh → use cached data
3. If stale/missing → fetch from API and cache it
4. Index in ChromaDB for vector search
5. Log: "Using cached news" or "Fetched X news articles"

### 4. Cache Metrics Tracking
**File:** `utils/cache_metrics.py`

Tracks:
- Price cache hits/misses
- News cache hits/misses
- Hit rates (percentage)
- Total requests

Usage:
```python
from utils.cache_metrics import get_cache_metrics

metrics = get_cache_metrics()
metrics.log_summary()
```

---

## Log Output Examples

### Successful Cache Hit (Second Request)
```
2025-11-11 19:45:10 - freshstart.coordinator - INFO - Starting analysis for AAPL
2025-11-11 19:45:10 - freshstart.coordinator - INFO - SQLite caching enabled
2025-11-11 19:45:10 - freshstart.storage - INFO - Cache HIT for AAPL prices (66 rows found)
2025-11-11 19:45:10 - freshstart.coordinator - INFO - Using cached price data for AAPL (66 days)
2025-11-11 19:45:10 - freshstart.coordinator - INFO - Fetched 8 fundamental metrics for AAPL
2025-11-11 19:45:10 - freshstart.storage - INFO - Cache HIT for AAPL news (50 articles found)
2025-11-11 19:45:10 - freshstart.coordinator - INFO - Using cached news for AAPL (50 articles)
2025-11-11 19:45:10 - freshstart.coordinator - INFO - ChromaDB vector store enabled
2025-11-11 19:45:11 - freshstart.coordinator - INFO - Indexed 50 news articles for AAPL
```

### Cache Miss (First Request)
```
2025-11-11 19:40:05 - freshstart.coordinator - INFO - Starting analysis for AAPL
2025-11-11 19:40:05 - freshstart.coordinator - INFO - SQLite caching enabled
2025-11-11 19:40:05 - freshstart.storage - INFO - Cache MISS for AAPL prices (no cached data)
2025-11-11 19:40:07 - freshstart.coordinator - INFO - Fetched 66 days of data for AAPL
2025-11-11 19:40:07 - freshstart.storage - INFO - Cached 66 price rows for AAPL (0 errors)
2025-11-11 19:40:07 - freshstart.coordinator - INFO - Fetched 8 fundamental metrics for AAPL
2025-11-11 19:40:09 - freshstart.storage - INFO - Cache MISS for AAPL news (no cached data)
2025-11-11 19:40:11 - freshstart.coordinator - INFO - Fetched 50 news articles for AAPL
2025-11-11 19:40:11 - freshstart.storage - INFO - Cached 50 news articles for AAPL (0 errors)
2025-11-11 19:40:11 - freshstart.coordinator - INFO - ChromaDB vector store enabled
2025-11-11 19:40:12 - freshstart.coordinator - INFO - Indexed 50 news articles for AAPL
```

---

## Testing the Implementation

### Option 1: Use Your Running Application

1. **Clear existing cache:**
   ```bash
   sqlite3 freshstart.db "DELETE FROM price_cache; DELETE FROM news_cache;"
   ```

2. **Make first request** (should see "Cache MISS" in logs):
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL", "user_tier": "basic"}'
   ```

3. **Make second request immediately** (should see "Cache HIT" in logs):
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL", "user_tier": "basic"}'
   ```

4. **Check logs:**
   ```bash
   tail -50 logs/freshstart.log
   ```

### Option 2: Run the Test Script

The `test_caching.py` script automates this process:

```bash
# Activate your virtual environment first
source venv/bin/activate  # or your venv path

# Run the test
python test_caching.py
```

Expected output:
- First request: ~5-10 seconds (fetching from APIs)
- Second request: <1 second (using cache)
- Speedup: 80-90% faster

### Option 3: Check Database Directly

```bash
# Count cached records
sqlite3 freshstart.db "SELECT ticker, COUNT(*) as count FROM price_cache GROUP BY ticker;"
sqlite3 freshstart.db "SELECT ticker, COUNT(*) as count FROM news_cache GROUP BY ticker;"

# Check cache freshness
sqlite3 freshstart.db "SELECT ticker, MAX(fetched_at) as last_fetch FROM price_cache GROUP BY ticker;"
```

---

## Configuration

### Cache Expiration
Default: **1 hour**

To change, modify the `max_age_hours` parameter in `coordinator/workflow.py`:

```python
# Price cache freshness check (line ~82)
if db.is_price_cache_fresh(ticker, max_age_hours=1):  # Change this

# News cache freshness check (line ~130)
if db.is_news_cache_fresh(ticker, max_age_hours=1):  # Change this
```

### Log Rotation
Default: **10MB file, 5 backups**

To change, modify `utils/logging_config.py`:

```python
setup_logging(
    name="freshstart",
    log_dir="logs",
    max_bytes=10 * 1024 * 1024,  # 10MB (change this)
    backup_count=5                # 5 backups (change this)
)
```

---

## Performance Impact

### Expected Cache Performance
- **Cache Hit:** <1 second (no API calls)
- **Cache Miss:** 5-10 seconds (API calls + database writes)
- **Speedup:** 80-90% faster on cache hits

### API Call Reduction
- **Before:** Every request = 3-4 API calls (price, fundamentals, news)
- **After:** Cached requests = 1 API call (only fundamentals)
- **Savings:** 75% reduction in API calls

---

## Files Modified

1. **New Files:**
   - `utils/logging_config.py` - Centralized logging configuration
   - `utils/cache_metrics.py` - Cache performance tracking
   - `test_caching.py` - Automated caching test script
   - `CACHING_AND_LOGGING.md` - This documentation

2. **Modified Files:**
   - `coordinator/workflow.py` - Added caching logic to `fetch_data()`
   - `storage/database.py` - Added logging + cache freshness methods
   - `api/main.py` - Updated to use centralized logging

---

## Troubleshooting

### Logs Not Appearing in File
- Check if `logs/` directory was created
- Verify permissions: `ls -la logs/`
- Check if `setup_logging()` was called in `api/main.py`

### Cache Not Working
- Verify database exists: `ls -la freshstart.db`
- Check cache contents: `sqlite3 freshstart.db "SELECT COUNT(*) FROM price_cache;"`
- Look for error messages in logs: `grep -i error logs/freshstart.log`

### Cache Always Misses
- Check cache staleness: `sqlite3 freshstart.db "SELECT ticker, fetched_at FROM price_cache LIMIT 5;"`
- Verify time is recent (within 1 hour)
- Check if `is_price_cache_fresh()` is being called

### Performance Not Improved
- Verify cache hits in logs: `grep "Cache HIT" logs/freshstart.log`
- Check if data is actually being cached: `sqlite3 freshstart.db "SELECT COUNT(*) FROM price_cache;"`
- Ensure you're making the same ticker request twice

---

## Monitoring Recommendations

### Daily Monitoring
```bash
# Check log file size
du -h logs/freshstart.log

# Count cache hits vs misses
grep "Cache HIT" logs/freshstart.log | wc -l
grep "Cache MISS" logs/freshstart.log | wc -l

# Check for errors
grep -i error logs/freshstart.log | tail -20
```

### Weekly Cleanup
```bash
# Remove old log backups (keeps last 5 automatically)
ls -lh logs/

# Clear stale cache (older than 7 days)
sqlite3 freshstart.db "DELETE FROM price_cache WHERE fetched_at < datetime('now', '-7 days');"
sqlite3 freshstart.db "DELETE FROM news_cache WHERE fetched_at < datetime('now', '-7 days');"
```

---

## Next Steps (Optional Enhancements)

1. **Add cache warming** - Pre-fetch popular tickers during off-hours
2. **Add cache statistics endpoint** - `/metrics` API endpoint showing cache performance
3. **Add distributed caching** - Redis for multi-instance deployments
4. **Add cache invalidation** - Manual trigger to refresh specific tickers
5. **Add alerting** - Email notifications for cache errors or API failures

---

**Last Updated:** 2025-11-11
**Version:** 1.0

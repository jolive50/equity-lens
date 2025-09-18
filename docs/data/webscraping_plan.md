# Web Scraping & Real-Time Harvesting Plan

## 1. Target Domains
- **Financial News Portals:** CNBC, Reuters, MarketWatch, Seeking Alpha (respect robots.txt & TOS).
- **Corporate Sources:** Investor relations pages for S&P 500 constituents, SEC press releases, exchange announcements.
- **Community Forums:** Reddit (r/stocks, r/investing), StockTwits public streams, specialized Discord communities (via bot agreements).
- **Alternative Sources:** Government data bulletins, Google Trends (API), professional blogs for niche coverage.

## 2. Compliance & Ethics
- Review each site's terms of service and robots.txt before scraping; prefer official APIs or RSS where available.
- Implement polite crawling (user-agent identification, rate limits, exponential backoff) and data retention aligned with legal guidance.
- Centralize consent tracking and contact points for site owners; provide opt-out mechanisms where required.

## 3. Scraper Architecture
- **Controller:** Orchestrate crawl schedules via Airflow DAGs or Prefect flows saved under `pipelines/realtime/webscrape_controller.py`.
- **Fetcher Layer:** Async HTTP clients (httpx) with rotating proxies and captcha mitigation where legally permissible.
- **Parser Layer:** Source-specific parsers stored in `webscraping/parsers/` transforming HTML/JSON into normalized article/event schemas.
- **Storage:** Raw payloads to `data/realtime/landing/web/` (compressed JSON). Cleaned text to `data/batch/processed/news/` for downstream NLP.

## 4. Tooling & Libraries
- **Libraries:** Playwright (dynamic pages), Newspaper3k (article extraction), BeautifulSoup4, trafilatura.
- **Quality Checks:** Duplicate detection (MinHash/SimHash), language detection, toxicity filtering.
- **Monitoring:** Use Grafana dashboards for crawl success rates, response times, and blocked requests.

## 5. Near-Term Action Items
1. Draft source registry (URL, cadence, credentials, legal status) under `webscraping/configs/source_catalog.yaml`.
2. Implement proof-of-concept scraper for one news site and one forum endpoint, storing outputs in landing zone.
3. Integrate fast sentiment scoring pipeline to convert scraped text into features for the sentiment agent.

"""Smart money data adapters for institutional flows, insider trading, and congressional disclosures."""
from __future__ import annotations

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InstitutionalHolding:
    """Institutional holding data structure."""
    institution_name: str
    shares_held: int
    market_value: float
    change_shares: int
    change_percent: float
    filing_date: str
    source: str


@dataclass
class InsiderTrade:
    """Insider trading data structure."""
    insider_name: str
    title: str
    transaction_type: str  # "buy" or "sell"
    shares: int
    price_per_share: float
    total_value: float
    transaction_date: str
    source: str


@dataclass
class CongressionalTrade:
    """Congressional trading disclosure data structure."""
    member_name: str
    chamber: str  # "House" or "Senate"
    transaction_type: str
    ticker: str
    amount_range: str
    transaction_date: str
    filing_date: str
    source: str


class SmartMoneyDataAdapter:
    """Base class for smart money data adapters."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Common request method with error handling."""
        try:
            params['apikey'] = self.api_key
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {endpoint}: {e}")
            return {}
    
    def get_institutional_holdings(self, ticker: str) -> List[InstitutionalHolding]:
        """Get institutional holdings for a ticker."""
        raise NotImplementedError
    
    def get_insider_trades(self, ticker: str) -> List[InsiderTrade]:
        """Get insider trades for a ticker."""
        raise NotImplementedError
    
    def get_congressional_trades(self, ticker: str) -> List[CongressionalTrade]:
        """Get congressional trades for a ticker."""
        raise NotImplementedError


class AlphaVantageSmartMoneyAdapter(SmartMoneyDataAdapter):
    """Alpha Vantage adapter for smart money data."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://www.alphavantage.co/query")
    
    def get_institutional_holdings(self, ticker: str) -> List[InstitutionalHolding]:
        """Get institutional holdings from Alpha Vantage."""
        data = self._make_request("", {
            "function": "INSTITUTIONAL_HOLDERS",
            "symbol": ticker
        })
        
        holdings = []
        if "institutional_holders" in data:
            for holder in data["institutional_holders"]:
                holdings.append(InstitutionalHolding(
                    institution_name=holder.get("name", ""),
                    shares_held=int(holder.get("shares", 0)),
                    market_value=float(holder.get("value", 0)),
                    change_shares=int(holder.get("change", 0)),
                    change_percent=float(holder.get("change_percent", 0)),
                    filing_date=holder.get("date", ""),
                    source="alpha_vantage"
                ))
        
        return holdings
    
    def get_insider_trades(self, ticker: str) -> List[InsiderTrade]:
        """Get insider trades from Alpha Vantage."""
        data = self._make_request("", {
            "function": "INSIDER_TRADES",
            "symbol": ticker
        })
        
        trades = []
        if "insider_trades" in data:
            for trade in data["insider_trades"]:
                trades.append(InsiderTrade(
                    insider_name=trade.get("insider_name", ""),
                    title=trade.get("title", ""),
                    transaction_type=trade.get("transaction_type", "").lower(),
                    shares=int(trade.get("shares", 0)),
                    price_per_share=float(trade.get("price", 0)),
                    total_value=float(trade.get("value", 0)),
                    transaction_date=trade.get("date", ""),
                    source="alpha_vantage"
                ))
        
        return trades
    
    def get_congressional_trades(self, ticker: str) -> List[CongressionalTrade]:
        """Alpha Vantage doesn't provide congressional trades."""
        return []


class FinnhubSmartMoneyAdapter(SmartMoneyDataAdapter):
    """Finnhub adapter for smart money data."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://finnhub.io/api/v1")
    
    def get_institutional_holdings(self, ticker: str) -> List[InstitutionalHolding]:
        """Get institutional holdings from Finnhub."""
        # Finnhub doesn't provide institutional holdings in free tier
        return []
    
    def get_insider_trades(self, ticker: str) -> List[InsiderTrade]:
        """Get insider trades from Finnhub."""
        data = self._make_request("/stock/insider-transactions", {
            "symbol": ticker,
            "from": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "to": datetime.now().strftime("%Y-%m-%d")
        })
        
        trades = []
        if isinstance(data, list):
            for trade in data:
                trades.append(InsiderTrade(
                    insider_name=trade.get("name", ""),
                    title=trade.get("title", ""),
                    transaction_type="buy" if trade.get("type", 0) > 0 else "sell",
                    shares=abs(trade.get("shares", 0)),
                    price_per_share=float(trade.get("price", 0)),
                    total_value=abs(trade.get("value", 0)),
                    transaction_date=trade.get("transactionDate", ""),
                    source="finnhub"
                ))
        
        return trades
    
    def get_congressional_trades(self, ticker: str) -> List[CongressionalTrade]:
        """Finnhub doesn't provide congressional trades."""
        return []


class CongressionalTradingAdapter:
    # What: Handle congressional trading disclosures once a real data provider is wired up.
    # Why: Keeps the service interface ready without returning fabricated trades.
    # How: Store an optional base URL and raise an error until an integration is complete.
    def __init__(self, *, data_source_url: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_source_url = data_source_url  # Data: URL string pointing at disclosure feed (or None).
    
    def get_congressional_trades(self, ticker: str) -> List[CongressionalTrade]:
        # What: Prevent returning placeholders for congressional trades.
        # Why: Real compliance-sensitive data must originate from official disclosures.
        # How: Raise a RuntimeError that describes which data feeds to integrate.
        # Data: Accepts the ticker symbol string; intentionally emits no trade records.
        raise RuntimeError(
            "Congressional trading data provider is not configured. "
            "Integrate a disclosure feed such as Capitol Trades, Senate Stock Watcher, "
            "or a licensed third-party dataset before requesting congressional activity."
        )
class SmartMoneyService:
    """Service that aggregates smart money data from multiple sources."""
    
    def __init__(
        self,
        adapters: Dict[str, SmartMoneyDataAdapter],
        *,
        congressional_adapter: Optional[CongressionalTradingAdapter] = None
    ):
        self.adapters = adapters
        # What: Hold a reference to the congressional disclosure adapter when one is provided.
        # Why: Some deployments may not have licensed congressional data yet, so we treat it as optional.
        # How: Store the object (or None) and let downstream methods respond accordingly.
        # Data: Expects an instance of CongressionalTradingAdapter or None when not configured.
        self.congressional_adapter = congressional_adapter
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_institutional_summary(self, ticker: str) -> Dict[str, Any]:
        """Get aggregated institutional holdings summary."""
        all_holdings = []
        
        for source, adapter in self.adapters.items():
            try:
                holdings = adapter.get_institutional_holdings(ticker)
                all_holdings.extend(holdings)
            except Exception as e:
                self.logger.error(f"Failed to get institutional holdings from {source}: {e}")
        
        if not all_holdings:
            return {
                "summary": "No recent institutional activity data available",
                "total_institutions": 0,
                "net_change": 0,
                "top_buyers": [],
                "top_sellers": []
            }
        
        # Calculate summary statistics
        total_institutions = len(set(h.institution_name for h in all_holdings))
        net_change = sum(h.change_shares for h in all_holdings)
        
        # Find top buyers and sellers
        buyers = [h for h in all_holdings if h.change_shares > 0]
        sellers = [h for h in all_holdings if h.change_shares < 0]
        
        top_buyers = sorted(buyers, key=lambda x: x.change_shares, reverse=True)[:3]
        top_sellers = sorted(sellers, key=lambda x: x.change_shares)[:3]
        
        # Generate summary text
        if net_change > 0:
            summary = f"Institutional net buying: {net_change:,} shares across {total_institutions} institutions"
        elif net_change < 0:
            summary = f"Institutional net selling: {abs(net_change):,} shares across {total_institutions} institutions"
        else:
            summary = f"Balanced institutional activity across {total_institutions} institutions"
        
        return {
            "summary": summary,
            "total_institutions": total_institutions,
            "net_change": net_change,
            "top_buyers": [
                {
                    "name": h.institution_name,
                    "shares": h.change_shares,
                    "value": h.market_value
                } for h in top_buyers
            ],
            "top_sellers": [
                {
                    "name": h.institution_name,
                    "shares": abs(h.change_shares),
                    "value": h.market_value
                } for h in top_sellers
            ]
        }
    
    def get_insider_summary(self, ticker: str) -> Dict[str, Any]:
        """Get aggregated insider trading summary."""
        all_trades = []
        
        for source, adapter in self.adapters.items():
            try:
                trades = adapter.get_insider_trades(ticker)
                all_trades.extend(trades)
            except Exception as e:
                self.logger.error(f"Failed to get insider trades from {source}: {e}")
        
        if not all_trades:
            return {
                "summary": "No recent insider trading activity",
                "total_trades": 0,
                "net_position": 0,
                "recent_activity": []
            }
        
        # Filter recent trades (last 90 days)
        recent_cutoff = datetime.now() - timedelta(days=90)
        recent_trades = [
            t for t in all_trades 
            if t.transaction_date and datetime.fromisoformat(t.transaction_date.replace('Z', '')) > recent_cutoff
        ]
        
        if not recent_trades:
            return {
                "summary": "No recent insider trading activity",
                "total_trades": 0,
                "net_position": 0,
                "recent_activity": []
            }
        
        # Calculate summary statistics
        total_trades = len(recent_trades)
        buy_trades = [t for t in recent_trades if t.transaction_type == "buy"]
        sell_trades = [t for t in recent_trades if t.transaction_type == "sell"]
        
        net_buy_shares = sum(t.shares for t in buy_trades) - sum(t.shares for t in sell_trades)
        
        # Generate summary text
        if net_buy_shares > 0:
            summary = f"Insider net buying: {net_buy_shares:,} shares in {total_trades} recent transactions"
        elif net_buy_shares < 0:
            summary = f"Insider net selling: {abs(net_buy_shares):,} shares in {total_trades} recent transactions"
        else:
            summary = f"Balanced insider activity: {total_trades} recent transactions"
        
        return {
            "summary": summary,
            "total_trades": total_trades,
            "net_position": net_buy_shares,
            "recent_activity": [
                {
                    "insider": t.insider_name,
                    "title": t.title,
                    "type": t.transaction_type,
                    "shares": t.shares,
                    "date": t.transaction_date
                } for t in recent_trades[:5]  # Top 5 recent trades
            ]
        }
    
    def get_congressional_summary(self, ticker: str) -> Dict[str, Any]:
        """Get congressional trading summary."""
        if not self.congressional_adapter:
            # What: Inform callers that congressional data is unavailable instead of fabricating output.
            # Why: Keeps the application transparent about missing datasets (compliance requirement).
            # How: Return a structured message with setup guidance rather than mock trades.
            # Data: Provides zero trades and a summary string that frontends can display verbatim.
            return {
                "summary": (
                    "Congressional trading analytics are unavailable because no disclosure data provider "
                    "is configured. Integrate a feed such as Capitol Trades or Senate Stock Watcher."
                ),
                "total_trades": 0,
                "recent_activity": []
            }
        
        try:
            trades = self.congressional_adapter.get_congressional_trades(ticker)
        except Exception as e:
            self.logger.error(f"Failed to get congressional trades: {e}")
            trades = []
        
        if not trades:
            return {
                "summary": "No recent congressional trading activity",
                "total_trades": 0,
                "recent_activity": []
            }
        
        # Filter recent trades (last 180 days)
        recent_cutoff = datetime.now() - timedelta(days=180)
        recent_trades = [
            t for t in trades 
            if t.transaction_date and datetime.fromisoformat(t.transaction_date) > recent_cutoff
        ]
        
        if not recent_trades:
            return {
                "summary": "No recent congressional trading activity",
                "total_trades": 0,
                "recent_activity": []
            }
        
        # Generate summary text
        total_trades = len(recent_trades)
        buy_trades = [t for t in recent_trades if t.transaction_type == "buy"]
        sell_trades = [t for t in recent_trades if t.transaction_type == "sell"]
        
        if len(buy_trades) > len(sell_trades):
            summary = f"Congressional net buying activity: {len(buy_trades)} buys vs {len(sell_trades)} sells"
        elif len(sell_trades) > len(buy_trades):
            summary = f"Congressional net selling activity: {len(sell_trades)} sells vs {len(buy_trades)} buys"
        else:
            summary = f"Balanced congressional activity: {total_trades} recent trades"
        
        return {
            "summary": summary,
            "total_trades": total_trades,
            "recent_activity": [
                {
                    "member": t.member_name,
                    "chamber": t.chamber,
                    "type": t.transaction_type,
                    "amount": t.amount_range,
                    "date": t.transaction_date
                } for t in recent_trades[:3]  # Top 3 recent trades
            ]
        }


def create_smart_money_service(api_keys: Dict[str, str]) -> SmartMoneyService:
    """Factory function to create smart money service."""
    adapters = {}
    
    if "alpha_vantage" in api_keys:
        adapters["alpha_vantage"] = AlphaVantageSmartMoneyAdapter(api_keys["alpha_vantage"])
    
    if "finnhub" in api_keys:
        adapters["finnhub"] = FinnhubSmartMoneyAdapter(api_keys["finnhub"])
    
    # What: Instantiate the service with whichever adapters are available right now.
    # Why: Some deployments may only provide a subset of smart-money data sources.
    # How: Pass the adapters dictionary and leave congressional data unplugged until a real feed is added.
    # Data: `congressional_adapter` is None by default; integrate a real provider before enabling that feature.
    return SmartMoneyService(adapters, congressional_adapter=None)

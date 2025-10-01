from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


class JsonFileRepository:
	"""Thread-safe JSON file repository for simple user-scoped data.

	Stores a single JSON map structure on disk: { "<user_id>": { ... } }
	"""

	def __init__(self, file_path: str) -> None:
		self._file_path = file_path
		self._lock = threading.Lock()
		dirname = os.path.dirname(self._file_path)
		if dirname and not os.path.exists(dirname):
			os.makedirs(dirname, exist_ok=True)
		# Initialize file if missing
		if not os.path.exists(self._file_path):
			self._write_all({})

	def _read_all(self) -> Dict[str, Any]:
		try:
			with open(self._file_path, "r", encoding="utf-8") as f:
				return json.load(f) or {}
		except FileNotFoundError:
			return {}

	def _write_all(self, data: Dict[str, Any]) -> None:
		with open(self._file_path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)

	def get_user_data(self, user_id: str) -> Any:
		with self._lock:
			data = self._read_all()
			return data.get(user_id)

	def set_user_data(self, user_id: str, value: Any) -> None:
		with self._lock:
			data = self._read_all()
			data[user_id] = value
			self._write_all(data)


class WatchlistRepository:
	"""Stores watchlists for users as a list of tickers."""

	def __init__(self, storage: JsonFileRepository) -> None:
		self._storage = storage

	def get(self, user_id: str) -> List[str]:
		return list(self._storage.get_user_data(user_id) or [])

	def add(self, user_id: str, ticker: str) -> List[str]:
		t = ticker.upper()
		current = set(self.get(user_id))
		current.add(t)
		result = sorted(current)
		self._storage.set_user_data(user_id, result)
		return result

	def remove(self, user_id: str, ticker: str) -> List[str]:
		t = ticker.upper()
		current = [x for x in self.get(user_id) if x != t]
		self._storage.set_user_data(user_id, current)
		return current

	def set_all(self, user_id: str, tickers: List[str]) -> List[str]:
		unique = []
		seen = set()
		for t in (tickers or []):
			u = (t or "").upper()
			if u and u not in seen:
				seen.add(u)
				unique.append(u)
		self._storage.set_user_data(user_id, unique)
		return unique


class HistoryRepository:
	"""Stores compact analysis history entries for users."""

	def __init__(self, storage: JsonFileRepository) -> None:
		self._storage = storage

	def list(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
		items = self._storage.get_user_data(user_id) or []
		return list(items)[-limit:]

	def append(self, user_id: str, entry: Dict[str, Any]) -> None:
		items = self._storage.get_user_data(user_id) or []
		items.append(entry)
		self._storage.set_user_data(user_id, items)


@dataclass
class AlertRule:
	"""Represents an alert threshold rule for a ticker.

	Fields:
	- ticker: symbol
	- condition: one of { "prob_down_gte", "prob_up_gte", "confidence_gte" }
	- threshold: float in [0,1]
	"""
	ticker: str
	condition: str
	threshold: float


class AlertsRepository:
	"""Stores alert rules for users by ticker."""

	def __init__(self, storage: JsonFileRepository) -> None:
		self._storage = storage

	def list(self, user_id: str) -> List[Dict[str, Any]]:
		return list(self._storage.get_user_data(user_id) or [])

	def upsert(self, user_id: str, rule: AlertRule) -> List[Dict[str, Any]]:
		rules = self.list(user_id)
		updated = []
		replaced = False
		for r in rules:
			if r.get("ticker") == rule.ticker and r.get("condition") == rule.condition:
				updated.append({"ticker": rule.ticker, "condition": rule.condition, "threshold": float(rule.threshold)})
				replaced = True
			else:
				updated.append(r)
		if not replaced:
			updated.append({"ticker": rule.ticker, "condition": rule.condition, "threshold": float(rule.threshold)})
		self._storage.set_user_data(user_id, updated)
		return updated

	def delete(self, user_id: str, ticker: str, condition: Optional[str] = None) -> List[Dict[str, Any]]:
		rules = self.list(user_id)
		remaining = []
		for r in rules:
			if r.get("ticker") != ticker.upper():
				remaining.append(r)
				continue
			if condition and r.get("condition") != condition:
				remaining.append(r)
		self._storage.set_user_data(user_id, remaining)
		return remaining



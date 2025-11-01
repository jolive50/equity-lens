"""JSON-based data persistence with file locking for durability.

This module provides thread-safe and process-safe repositories for storing
user data in JSON files. Designed with SOLID principles and comprehensive
explanations for college-level understanding.

What this module does:
- Stores user watchlists, history, and alert rules in JSON files
- Provides thread-safe operations using locks
- Implements file locking to prevent corruption from concurrent writes
- Handles file I/O errors gracefully

Why we need this:
- Simple persistence without database complexity
- Safe concurrent access from multiple threads/processes
- Easy to backup and debug (human-readable JSON)

How it works:
- JsonFileRepository: Base class handling file I/O with locking
- WatchlistRepository: Manages user stock watchlists
- HistoryRepository: Stores analysis history
- AlertsRepository: Manages price/confidence alert rules

College-Level Concepts:
- File Locking: Prevents race conditions when multiple processes write
- Thread Safety: Using locks to coordinate access within a process
- Repository Pattern: Abstraction layer over data storage
- SOLID Principles: Single Responsibility, Interface Segregation
"""
from __future__ import annotations

import json  # For JSON serialization/deserialization
import logging  # For error and debug logging
import os  # For file system operations
import threading  # For thread-level locks
import time  # For retry delays
from dataclasses import dataclass  # For simple data classes
from typing import Any, Dict, List, Optional

# File locking library
# What: Provides cross-platform file locking (Windows and Unix)
# Why: Prevents corruption when multiple processes access the same file
# How: Implements platform-specific locking mechanisms
try:
    import portalocker  # pip install portalocker
    HAS_PORTALOCKER = True
except ImportError:
    # Fallback if portalocker not installed
    # What: Allow code to run without locking if library missing
    # Why: Graceful degradation (works but less safe)
    # How: Set flag to disable file locking
    HAS_PORTALOCKER = False

logger = logging.getLogger(__name__)


class JsonFileRepository:
	"""Thread-safe and process-safe JSON file repository for user-scoped data.

	What this does: Manages a JSON file storing user data as {"user_id": {...}}
	Why: Simple persistence layer without database complexity
	How: Uses threading locks (within process) and file locks (across processes)

	This class demonstrates SOLID principles:
	- Single Responsibility: Only handles JSON file I/O with locking
	- Open/Closed: Subclasses can extend behavior without modifying this class
	- Interface Segregation: Clean, focused interface (_read_all, _write_all)
	- Dependency Inversion: Clients depend on this abstraction, not file system

	Thread Safety (same process):
	- Uses threading.Lock() to coordinate access within a single process
	- Prevents race conditions when multiple threads read/write

	Process Safety (different processes):
	- Uses portalocker for file locking across processes
	- Prevents corruption when multiple StockSense instances run simultaneously
	- Example: Web server + background worker both accessing same file

	College-Level Analogy:
	Imagine a shared notebook in a library. The threading lock is like a person
	holding the notebook (others wait). The file lock is like a checkout system
	that prevents multiple people from taking it home simultaneously.
	"""

	def __init__(
		self,
		file_path: str,  # Path to JSON file
		*,
		max_retries: int = 3,  # Retries if file locked
		retry_delay: float = 0.1,  # Seconds between retries
	) -> None:
		"""Initialize repository with file path and retry configuration.

		What: Sets up repository for a specific JSON file
		Why: Each repository manages one file (watchlists, history, alerts)
		How: Creates directory structure, initializes empty file if needed

		Args:
			file_path: Path to JSON file (e.g., "data/watchlists.json")
			max_retries: How many times to retry if file locked
			retry_delay: How long to wait between retries (seconds)

		Thread Safety:
		- threading.Lock() ensures only one thread accesses file at a time
		- Prevents: Thread A reading while Thread B is writing (corrupt data)

		Process Safety:
		- portalocker ensures only one process accesses file at a time
		- Prevents: Process A and B both writing simultaneously (file corruption)
		"""
		self._file_path = file_path
		self._lock = threading.Lock()  # Thread-level lock (within process)
		self._max_retries = max_retries
		self._retry_delay = retry_delay

		# Warn if file locking not available
		# What: Alert user if portalocker not installed
		# Why: Code will work but less safe in multi-process scenarios
		# How: Check HAS_PORTALOCKER flag
		if not HAS_PORTALOCKER:
			logger.warning(
				"portalocker not installed. File locking disabled. "
				"Install with: pip install portalocker"
			)

		# Create directory structure
		# What: Ensure parent directories exist
		# Why: Can't create file if directory doesn't exist
		# How: os.makedirs with exist_ok=True (doesn't error if exists)
		dirname = os.path.dirname(self._file_path)
		if dirname and not os.path.exists(dirname):
			os.makedirs(dirname, exist_ok=True)

		# Initialize file if missing
		# What: Create empty JSON file if doesn't exist
		# Why: Avoid FileNotFoundError on first access
		# How: Write empty dict ({}) to file
		if not os.path.exists(self._file_path):
			self._write_all({})
			logger.info(f"Initialized new repository file: {self._file_path}")

	def _read_all(self) -> Dict[str, Any]:
		"""Read entire JSON file with file locking.

		What: Loads complete JSON file into memory
		Why: Need current state to read/modify user data
		How: Opens file with shared lock (multiple readers OK), parses JSON

		Returns:
			Dictionary of all user data {"user_id": {...}, ...}

		Locking Strategy:
		- Uses LOCK_SH (shared lock) if available
		- Allows multiple concurrent readers (reading is safe)
		- Blocks if any process has exclusive lock (writing)

		Error Handling:
		- FileNotFoundError: Returns empty dict (file created on write)
		- JSONDecodeError: Logs error, returns empty dict (corruption recovery)
		- Retries if file locked by another process
		"""
		# Try multiple times if file locked
		# What: Retry loop with exponential backoff
		# Why: Another process might have file locked temporarily
		# How: Sleep and retry up to max_retries times
		for attempt in range(self._max_retries):
			try:
				# Open file with shared lock (read lock)
				# What: Opens file for reading with file-level lock
				# Why: Ensures consistent read (no one writing simultaneously)
				# How: portalocker.Lock opens and locks atomically
				if HAS_PORTALOCKER:
					# With file locking (safer)
					with portalocker.Lock(
						self._file_path,
						mode='r',
						encoding='utf-8',
						flags=portalocker.LOCK_SH,  # Shared lock (read)
						timeout=self._retry_delay  # How long to wait for lock
					) as f:
						data = json.load(f)
						return data if data else {}
				else:
					# Without file locking (fallback)
					# What: Simple file open without locking
					# Why: Fallback when portalocker not available
					# How: Standard Python file I/O
					with open(self._file_path, 'r', encoding='utf-8') as f:
						data = json.load(f)
						return data if data else {}

			except FileNotFoundError:
				# File doesn't exist yet
				# What: Handle missing file gracefully
				# Why: First access might be before file created
				# How: Return empty dict, file will be created on write
				logger.debug(f"File not found: {self._file_path}, returning empty dict")
				return {}

			except json.JSONDecodeError as e:
				# Corrupted JSON
				# What: Handle invalid JSON (corruption or manual edit)
				# Why: Corrupt data shouldn't crash the app
				# How: Log error, return empty dict (data loss but app continues)
				logger.error(f"JSON decode error in {self._file_path}: {e}")
				logger.error("File may be corrupted. Returning empty dict.")
				return {}

			except portalocker.exceptions.LockException as e:
				# File locked by another process
				# What: Another process holds the lock
				# Why: Concurrent access attempting
				# How: Retry after delay
				if attempt < self._max_retries - 1:
					logger.debug(f"File locked, retrying ({attempt + 1}/{self._max_retries})...")
					time.sleep(self._retry_delay * (attempt + 1))  # Exponential backoff
					continue
				else:
					# Max retries exceeded
					# What: Couldn't get lock after all retries
					# Why: File held by long-running operation or deadlock
					# How: Raise error (caller should handle)
					logger.error(f"Failed to read {self._file_path} after {self._max_retries} retries")
					raise RuntimeError(f"Could not acquire read lock on {self._file_path}") from e

			except Exception as e:
				# Unexpected error
				# What: Catch any other errors
				# Why: Don't crash on unexpected issues
				# How: Log and return empty dict
				logger.error(f"Unexpected error reading {self._file_path}: {e}", exc_info=True)
				return {}

		# Should not reach here, but return empty dict as fallback
		return {}

	def _write_all(self, data: Dict[str, Any]) -> None:
		"""Write entire JSON file with exclusive file locking.

		What: Overwrites entire JSON file with new data
		Why: Persist changes to disk
		How: Opens file with exclusive lock, writes JSON

		Args:
			data: Complete user data dictionary to write

		Locking Strategy:
		- Uses LOCK_EX (exclusive lock) if available
		- Blocks all other readers and writers
		- Ensures atomic write (no partial/corrupt data)

		Safety Features:
		- Write to temporary file first, then rename (atomic operation)
		- Ensures file is never in invalid state
		- If crash during write, original file unchanged

		Why Atomic Writes:
		Without atomicity:
		1. Open file for write (truncates file to 0 bytes)
		2. Process crashes
		3. Data lost forever!

		With atomicity:
		1. Write to temp file
		2. Rename temp to target (atomic operation on most systems)
		3. If crash before rename, original file intact
		"""
		# Try multiple times if file locked
		for attempt in range(self._max_retries):
			try:
				# Write to temporary file first (atomic write pattern)
				# What: Write to temp file, then rename to target
				# Why: Prevents corruption if write interrupted
				# How: Create .tmp file, write, rename
				temp_path = f"{self._file_path}.tmp"

				if HAS_PORTALOCKER:
					# With file locking (safer)
					with portalocker.Lock(
						temp_path,
						mode='w',
						encoding='utf-8',
						flags=portalocker.LOCK_EX,  # Exclusive lock (write)
						timeout=self._retry_delay
					) as f:
						json.dump(data, f, ensure_ascii=False, indent=2)

				else:
					# Without file locking (fallback)
					with open(temp_path, 'w', encoding='utf-8') as f:
						json.dump(data, f, ensure_ascii=False, indent=2)

				# Atomic rename
				# What: Replace target file with temp file
				# Why: Atomic operation (all or nothing)
				# How: os.replace is atomic on Windows and Unix
				os.replace(temp_path, self._file_path)

				# Success!
				return

			except portalocker.exceptions.LockException as e:
				# File locked by another process
				if attempt < self._max_retries - 1:
					logger.debug(f"File locked during write, retrying ({attempt + 1}/{self._max_retries})...")
					time.sleep(self._retry_delay * (attempt + 1))  # Exponential backoff
					continue
				else:
					# Max retries exceeded
					logger.error(f"Failed to write {self._file_path} after {self._max_retries} retries")
					raise RuntimeError(f"Could not acquire write lock on {self._file_path}") from e

			except Exception as e:
				# Unexpected error
				# What: Handle any other errors
				# Why: Don't lose data silently
				# How: Log error and raise (caller should handle)
				logger.error(f"Error writing {self._file_path}: {e}", exc_info=True)
				raise RuntimeError(f"Failed to write repository file: {e}") from e

		# Should not reach here
		raise RuntimeError(f"Failed to write {self._file_path} after all retries")

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



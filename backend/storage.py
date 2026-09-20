from typing import Dict, List, Any, Optional
import pandas as pd

# Simple in-memory storage for dataframes and audit metadata
DATAFRAMES: Dict[str, pd.DataFrame] = {}
DATASET_HISTORY: Dict[str, List[pd.DataFrame]] = {}
PLAN_CACHE: Dict[str, Dict[str, Any]] = {}
AUDIT_LOG: List[Dict[str, Any]] = []


def snapshot_dataframe(file_id: str) -> None:
	"""Store a deep copy that can be used for undo or diffing."""
	if file_id in DATAFRAMES:
		DATASET_HISTORY.setdefault(file_id, []).append(DATAFRAMES[file_id].copy(deep=True))


def get_last_snapshot(file_id: str) -> Optional[pd.DataFrame]:
	"""Return the most recent snapshot if present."""
	history = DATASET_HISTORY.get(file_id, [])
	if history:
		return history[-1]
	return None


def register_plan(plan_id: str, payload: Dict[str, Any]) -> None:
	PLAN_CACHE[plan_id] = payload


def get_plan(plan_id: str) -> Dict[str, Any] | None:
	return PLAN_CACHE.get(plan_id)


def discard_plan(plan_id: str) -> None:
	PLAN_CACHE.pop(plan_id, None)


def append_audit_entry(entry: Dict[str, Any]) -> None:
	AUDIT_LOG.append(entry)

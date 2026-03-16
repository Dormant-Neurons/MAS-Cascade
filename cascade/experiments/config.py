"""
Experiment configuration loader utilities.

Supports JSON and YAML configuration files containing either a single experiment
definition or a batch under the keys ``configs``/``runs``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

CONFIG_LIST_KEYS = ("configs", "runs", "batch")


def _load_yaml(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "ERROR: YAML support requires PyYAML. Install with `pip install pyyaml` "
            "or provide a JSON configuration file."
        ) from exc
    data = yaml.safe_load(text)
    return data if data is not None else {}


def load_config_file(path: str | Path) -> Any:
    """Load a JSON or YAML configuration file and return the raw payload."""
    file_path = Path(path)
    if not file_path.exists():
        raise SystemExit(f"ERROR: config file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()
    if suffix in {".json"}:
        return json.loads(text or "{}")
    if suffix in {".yml", ".yaml"}:
        return _load_yaml(text)

    # Try JSON first, then YAML as fallback
    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError:
        return _load_yaml(text)


def normalize_configs(payload: Any) -> List[Dict[str, Any]]:
    """
    Normalize a configuration payload into a list of per-run dictionaries.

    The payload can take one of several shapes:
      * dict with ``configs``/``runs``/``batch`` (list of run dicts)
      * dict describing a single run
      * list of run dicts
    """
    if payload is None:
        return [{}]

    if isinstance(payload, list):
        configs = payload
        defaults: Dict[str, Any] = {}
    elif isinstance(payload, dict):
        configs = None
        defaults = dict(payload.get("defaults", {}))
        for key in CONFIG_LIST_KEYS:
            if key in payload:
                configs = payload[key]
                break
        if configs is None:
            configs = [payload]
    else:
        raise SystemExit("ERROR: configuration payload must be a dict or list.")

    if not isinstance(configs, Iterable):
        raise SystemExit("ERROR: configuration list must be iterable.")

    normalized: List[Dict[str, Any]] = []
    for entry in configs:
        if entry is None:
            entry = {}
        if not isinstance(entry, dict):
            raise SystemExit("ERROR: each configuration entry must be a mapping/dict.")
        run_cfg: Dict[str, Any] = dict(defaults)
        run_cfg.update(entry)
        normalized.append(run_cfg)

    return normalized


def load_and_normalize(path: str | Path) -> List[Dict[str, Any]]:
    """Convenience helper combining load and normalize steps."""
    return normalize_configs(load_config_file(path))

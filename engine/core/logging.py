from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "run_id": self.run_id,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Attach structured extras if present
        extras = getattr(record, "extra", None)
        if isinstance(extras, dict) and extras:
            base["extra"] = extras

        # Attach exception info if present
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(base, ensure_ascii=True)


def setup_logging(run_id: str, level: str = "INFO") -> None:
    """
    Configure root logger to emit one JSON line per log record to stdout.

    This is intentionally minimal (no 3rd-party deps) and deterministic in schema.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter(run_id=run_id))
    root.addHandler(handler)

    # Reduce noise from common libraries (adjust as you like)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("psycopg").setLevel(logging.WARNING)
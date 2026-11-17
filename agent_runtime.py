"""Runtime telemetry and time budgets shared by agent middleware and tools."""
from contextvars import ContextVar
import json
import logging
import sqlite3
import time
import httpx


LOGGER = logging.getLogger("uvicorn.error.agent_runtime")
tool_deadline: ContextVar[float | None] = ContextVar("agent_tool_deadline", default=None)


def deadline_for(config, seconds):
    started = config.get("turn_started")
    return started + seconds if isinstance(started, (int, float)) else None


def remaining(deadline):
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def bounded_timeout(budget, configured):
    if isinstance(configured, httpx.Timeout):
        phases = {phase: min(budget, value) if value is not None else budget
                  for phase, value in configured.as_dict().items()}
        return httpx.Timeout(**phases)
    return min(budget, configured) if configured is not None else budget


def transient_io(error):
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if isinstance(error, sqlite3.OperationalError):
        code = getattr(error, "sqlite_errorcode", 0) or 0
        return (code & 0xff) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
    return False


def emit(config, event, **fields):
    turn_id = config.get("turn_id")
    if not turn_id:
        return
    record = {"turn_id": turn_id, "event": event, **fields}
    LOGGER.info("agent_runtime %s", json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                extra={"agent_event": record})

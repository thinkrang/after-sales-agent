"""Bounded in-process locks: serialize one conversation, not all LLM I/O.

Database constraints remain the cross-process safety boundary. These locks do
not claim distributed checkpoint serialization across multiple API workers.
"""
import hashlib
from threading import Lock

_locks = {namespace: [Lock() for _ in range(256)] for namespace in ("dialogue", "execution")}


def conversation_lock(key: str):
    namespace = "execution" if key.startswith("execution:") else "dialogue"
    locks = _locks[namespace]
    slot = int.from_bytes(hashlib.sha256(key.encode()).digest()[:2], "big") % len(locks)
    return locks[slot]

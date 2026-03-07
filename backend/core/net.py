"""
bakup/core/net.py
─────────────────────────────────────────────────────────────────────────────
Network helpers — port availability check and auto-detection.

Used by both the dev entry point (main.py) and the compiled entry point
(bakup_server.py) to avoid startup failures when the default port is busy.
"""

import socket


def is_port_available(host: str, port: int) -> bool:
    """Return True if *port* on *host* can be bound (i.e. is free).

    Does NOT set SO_REUSEADDR — on Windows that flag allows "port stealing",
    which would make busy ports appear free.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_free_port(host: str = "127.0.0.1", start: int = 8000,
                   max_attempts: int = 20) -> int:
    """Return *start* if it is free, otherwise scan upward until a free port
    is found.  Raises ``RuntimeError`` after *max_attempts* failures.

    Scans sequentially: 8000 → 8001 → … → 8019 (by default).
    """
    for offset in range(max_attempts):
        candidate = start + offset
        if is_port_available(host, candidate):
            return candidate
    raise RuntimeError(
        f"bakup: no free port found in range {start}–{start + max_attempts - 1}"
    )


def resolve_port(host: str, preferred: int) -> int:
    """Check *preferred* first; if busy, find the next free port and log the
    change.  Returns the port to use."""
    if is_port_available(host, preferred):
        return preferred

    actual = find_free_port(host, start=preferred + 1)
    print(
        f"bakup: port {preferred} is in use — falling back to {actual}"
    )
    return actual

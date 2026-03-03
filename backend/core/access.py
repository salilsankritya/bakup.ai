"""
bakup/core/access.py
─────────────────────────────────────────────────────────────────────────────
Startup access key check.

PURPOSE
    Prevents accidental redistribution of the Developer Preview binary by
    requiring the operator to supply a known access key via environment variable.
    This is not a cryptographic security boundary and is not DRM.
    The hash is visible in source. A determined person can bypass it.
    That is accepted. The goal is friction against casual unauthorised use,
    and a clear signal to the operator that they need a valid key.

MECHANISM
    1. Read BAKUP_ACCESS_KEY from the environment.
    2. Compute SHA-256(key.strip()) using Python's stdlib hashlib — no deps.
    3. Compare result against EXPECTED_KEY_HASH with hmac.compare_digest
       (constant-time, avoids trivial timing side-channels).
    4. On failure: print a clear, actionable message to stderr and call
       sys.exit(1) before any service starts.

KEY ROTATION
    Run:  python scripts/generate_key_hash.py <new-key>
    Then: replace EXPECTED_KEY_HASH below with the output.
"""

import hashlib
import hmac
import os
import sys
import textwrap

# SHA-256 hex digest of the valid access key.
# Generated via: python scripts/generate_key_hash.py tango
EXPECTED_KEY_HASH: str = "7063d51d1b2da165eee042de5d33cc27281ea80e1a291488c903b7fb5fc31da7"

# Environment variable the operator must set.
ENV_VAR: str = "BAKUP_ACCESS_KEY"


def _hash_key(raw: str) -> str:
    """Return the SHA-256 hex digest of a stripped key string."""
    return hashlib.sha256(raw.strip().encode("utf-8")).hexdigest()


def _keys_match(provided_hash: str, expected_hash: str) -> bool:
    """Constant-time comparison. Both arguments must be equal-length hex strings."""
    # hmac.compare_digest requires the same type; encode to bytes for safety.
    return hmac.compare_digest(
        provided_hash.encode("ascii"),
        expected_hash.encode("ascii"),
    )


def check_access_key() -> None:
    """
    Validate the access key at startup.

    Call this once, as early as possible in the application entry point,
    before any routes, database connections, or background tasks are started.

    Exits the process with code 1 on any failure. Never raises.
    """
    raw_key = os.environ.get(ENV_VAR, "")

    if not raw_key.strip():
        _fail_missing()

    provided_hash = _hash_key(raw_key)

    if not _keys_match(provided_hash, EXPECTED_KEY_HASH):
        _fail_invalid()

    # Valid. No output on success — standard unix convention.


# ── Failure handlers ──────────────────────────────────────────────────────────

def _fail_missing() -> None:
    _exit(
        title="Access key not set.",
        body=f"""\
        Set the {ENV_VAR} environment variable before starting Bakup.

        If you received an access key with your Developer Preview download,
        add it to your .env file:

            {ENV_VAR}=<your-key>

        If you do not have a key, request one at: https://bakup.ai
        """,
    )


def _fail_invalid() -> None:
    _exit(
        title="Access key is invalid.",
        body=f"""\
        The value provided for {ENV_VAR} does not match a known key.

        Double-check your .env file for typos or trailing whitespace.
        Keys are case-sensitive.

        If your key is not working, contact the team that issued it.
        """,
    )


def _exit(title: str, body: str) -> None:
    border = "─" * 60
    formatted_body = textwrap.dedent(body).strip()
    message = (
        f"\n{border}\n"
        f"  bakup: {title}\n"
        f"{border}\n\n"
        f"  {formatted_body.replace(chr(10), chr(10) + '  ')}\n\n"
        f"{border}\n"
    )
    print(message, file=sys.stderr)
    sys.exit(1)

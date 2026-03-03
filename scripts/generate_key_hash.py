#!/usr/bin/env python3
"""
scripts/generate_key_hash.py
─────────────────────────────────────────────────────────────────────────────
Utility for rotating the Developer Preview access key.

Usage:
    python scripts/generate_key_hash.py <new-access-key>

Output:
    The SHA-256 hex digest of the key, ready to paste into:
        backend/core/access.py → EXPECTED_KEY_HASH

This script has no dependencies beyond Python stdlib.
Requires Python 3.6+.
"""

import hashlib
import sys


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python scripts/generate_key_hash.py <access-key>", file=sys.stderr)
        sys.exit(1)

    key = sys.argv[1].strip()
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()

    border = "─" * 60
    print(f"\n{border}")
    print(f"  Key     : {key}")
    print(f"  SHA-256 : {digest}")
    print(f"{border}")
    print()
    print("  Paste into backend/core/access.py → EXPECTED_KEY_HASH:\n")
    print(f'  EXPECTED_KEY_HASH: str = "{digest}"')
    print()
    print("  Then update BAKUP_ACCESS_KEY in your .env file.\n")


if __name__ == "__main__":
    main()

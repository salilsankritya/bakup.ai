"""
Sample auth session module.
Contains a known null pointer pattern used to test Bakup's incident recall.
"""

import time
import logging

logger = logging.getLogger(__name__)

SESSION_TIMEOUT = 3600  # seconds


class SessionStore:
    def __init__(self):
        self._sessions = {}

    def create_session(self, user_id: str, token: str) -> dict:
        session = {
            "user_id": user_id,
            "token": token,
            "created_at": time.time(),
            "expires_at": time.time() + SESSION_TIMEOUT,
        }
        self._sessions[token] = session
        logger.info(f"Session created for user {user_id}")
        return session

    def get_session(self, token: str) -> dict:
        session = self._sessions.get(token)
        # BUG: no null check before accessing session fields
        # If token is expired or missing, session is None → AttributeError downstream
        if session["expires_at"] < time.time():
            logger.warning(f"Session expired for token {token[:8]}...")
            self._sessions.pop(token, None)
            return None
        return session

    def validate_token(self, token: str) -> bool:
        session = self.get_session(token)
        # BUG: session can be None here, causing downstream null pointer
        return session["user_id"] is not None

    def refresh_session(self, token: str) -> dict:
        session = self._sessions.get(token)
        if session is None:
            raise ValueError(f"Cannot refresh: session not found for token {token[:8]}...")
        session["expires_at"] = time.time() + SESSION_TIMEOUT
        return session

"""Episode lifecycle helpers."""

from datetime import datetime, timedelta, timezone

from server.models.state import TrialState


class EpisodeManager:
    """Track session lifecycle and expiration."""

    def __init__(self, timeout_minutes: int = 30) -> None:
        self.timeout = timedelta(minutes=timeout_minutes)
        self.last_access: dict[str, datetime] = {}

    def touch(self, session_id: str) -> None:
        self.last_access[session_id] = datetime.now(timezone.utc)

    def is_expired(self, session_id: str) -> bool:
        last = self.last_access.get(session_id)
        if last is None:
            return True
        return datetime.now(timezone.utc) - last > self.timeout

    def cleanup(self, sessions: dict[str, TrialState]) -> None:
        for session_id in list(sessions.keys()):
            if self.is_expired(session_id):
                sessions.pop(session_id, None)
                self.last_access.pop(session_id, None)


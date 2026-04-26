"""Thin HTTP client wrapper for OpenEnv-compatible training loops."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class EnvClient:
    base_url: str
    timeout: float = 60.0

    def wait_until_ready(self, max_wait_seconds: int = 120, poll_interval_seconds: int = 5) -> None:
        deadline = time.time() + max_wait_seconds
        while time.time() < deadline:
            try:
                response = httpx.get(self.base_url + "/", timeout=5)
                if response.status_code < 500:
                    return
            except (httpx.ConnectError, httpx.ConnectTimeout):
                pass
            time.sleep(poll_interval_seconds)
        raise RuntimeError(
            f"Environment server not reachable at {self.base_url} after {max_wait_seconds}s. "
            "Start the server before running GRPO."
        )

    def reset(self, task_id: str, seed: int | None = None) -> dict[str, Any]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.post("/reset", json={"task_id": task_id, "seed": seed})
            response.raise_for_status()
            return response.json()

    def step(self, session_id: str, action: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.post("/step", json={"session_id": session_id, "action": action})
            response.raise_for_status()
            return response.json()

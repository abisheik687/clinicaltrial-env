"""Protocol loader and validator."""

from __future__ import annotations

from pathlib import Path

import yaml

from server.data.schemas.protocol_schema import TrialProtocol


class ProtocolLoader:
    """Load trial protocols from YAML files."""

    def __init__(self, protocol_dir: Path) -> None:
        self.protocol_dir = protocol_dir
        self._cache: dict[str, TrialProtocol] = {}

    def load(self, filename: str) -> TrialProtocol:
        """Load and validate a single protocol file."""
        if filename in self._cache:
            return self._cache[filename]
        path = self.protocol_dir / filename
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        protocol = TrialProtocol.model_validate(data)
        self._cache[filename] = protocol
        return protocol

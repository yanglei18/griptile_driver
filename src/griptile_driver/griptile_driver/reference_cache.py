"""Cross-session unloaded references on the acquisition host."""

from hashlib import sha256
import json
import os
from pathlib import Path

from griptile_core.reference_file import load_reference_file, save_json_atomic


class ReferenceCache:
    def __init__(self):
        state = os.environ.get("XDG_STATE_HOME")
        root = Path(state) if state and Path(state).is_absolute() else Path.home() / ".local" / "state"
        self.directory = root / "griptile_driver" / "references"

    def _path(self, binding, direction):
        key = json.dumps([binding, direction], sort_keys=True, separators=(",", ":")).encode()
        return self.directory / (sha256(key).hexdigest() + ".json")

    def load(self, binding, direction):
        try:
            return load_reference_file(self._path(binding, direction), binding, direction)
        except FileNotFoundError:
            return None

    def save(self, binding, direction, profile):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        save_json_atomic(self._path(binding, direction), {
            "version": 2, "binding": binding, "direction": direction, "reference": profile.to_dict(),
        })

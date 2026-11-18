"""Local, device-bound adoption of a validated spatial display calibration."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import time

from pyqtgraph.Qt import QtCore

from griptile_core.equalization import EqualizationProfile
from griptile_core.protocol import PROTOCOL_ID
from .storage import save_json_atomic


_UNREAD = object()


@dataclass(frozen=True)
class _Binding:
    sensor_id: str
    hardware_id: str
    protocol_id: str
    layout_id: str
    direction: str
    reference_digest: str

    @classmethod
    def from_context(cls, context):
        return cls(context.sensor_id, context.hardware_id, PROTOCOL_ID, context.layout_id,
                   context.direction, context.reference_digest)

    def filename(self):
        encoded = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return sha256(encoded).hexdigest() + ".json"


def _file_version(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


class EqualizationSettings:
    """Only apply gains on their matching live device, direction and zero reference."""

    def __init__(self, model, directory=None):
        self.model = model
        if directory is None:
            root = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.GenericConfigLocation)
            directory = Path(root) / "griptile_driver" / "equalization"
        self.directory = Path(directory)
        self.profile = None
        self.enabled = False
        self.error = ""
        self._context = self._binding = self._version = None
        self._next_check = 0.

    @property
    def persistent(self):
        return self._binding is not None

    @property
    def active(self):
        return self.profile if self.enabled else None

    def sync(self):
        context = self.model.equalization_context
        if context is not self._context:
            self._context = context
            self.profile, self.enabled, self.error = None, False, ""
            self._version = None
            self._binding = (_Binding.from_context(context) if context and context.hardware_id else None)
            self._next_check = 0.
        if not self.persistent or time.monotonic() < self._next_check:
            return
        self._next_check = time.monotonic() + 1.
        path = self.directory / self._binding.filename()
        try:
            version = _file_version(path)
            if version == self._version:
                return
            self._version = version
            if version is None:
                self.profile, self.enabled, self.error = None, False, ""
                return
            with path.open("rb") as stream:
                payload = stream.read(131073)
            if len(payload) > 131072:
                raise ValueError("标定文件过大")
            doc = json.loads(payload)
            if (not isinstance(doc, dict) or type(doc.get("version")) is not int or doc["version"] != 1
                    or doc.get("binding") != asdict(self._binding)
                    or type(doc.get("enabled")) is not bool):
                raise ValueError("标定文件版本或设备绑定不符")
            profile = EqualizationProfile.from_dict(doc.get("profile"))
            if profile.direction != context.direction:
                raise ValueError("标定方向与当前采集不符")
            self.profile, self.enabled, self.error = profile, doc["enabled"], ""
        except (ValueError, OSError, TypeError, OverflowError, RecursionError) as error:
            self.profile, self.enabled = None, False
            self.error = f"无法读取阵元标定：{error}"
            if isinstance(error, OSError):
                # Retry a failed stat/open even when the file version is unchanged.
                self._version = _UNREAD

    def _replace(self, profile, enabled):
        if self._context is None:
            raise ValueError("等待当前设备和有效的无载参考")
        if self.persistent:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / self._binding.filename()
            lock = QtCore.QLockFile(str(path) + ".lock")
            if not lock.tryLock(0):
                raise ValueError("另一窗口正在更新阵元标定，请稍后重试")
            try:
                if _file_version(path) != self._version:
                    self._next_check = 0.
                    self.sync()
                    raise ValueError("阵元标定已在另一窗口更新，请确认后重试")
                if profile is None:
                    path.unlink(missing_ok=True)
                else:
                    save_json_atomic(path, {"version": 1, "binding": asdict(self._binding),
                                           "enabled": enabled, "profile": profile.to_dict()})
                self._version = _file_version(path)
            finally:
                lock.unlock()
        self.profile, self.enabled, self.error = profile, enabled, ""

    def adopt(self, profile, context):
        self.sync()
        if (context is None or self._context != context or not isinstance(profile, EqualizationProfile)
                or profile.direction != context.direction):
            raise ValueError("设备、会话或无载参考已改变，请重新执行阵元标定")
        self._replace(profile, True)

    def set_enabled(self, enabled):
        self.sync()
        if self.profile is None:
            raise ValueError("当前设备尚无通过验证的阵元标定")
        self._replace(self.profile, bool(enabled))

    def clear(self):
        self.sync()
        self._replace(None, False)

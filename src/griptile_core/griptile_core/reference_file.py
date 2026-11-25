"""Bounded reference-file decoding and atomic local JSON writes."""

import json
import os
from pathlib import Path
import tempfile

from .display_response import ReferenceProfile


def save_json_atomic(path, document):
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=destination.parent, encoding="utf-8",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_reference_file(path, binding, direction, *, allow_legacy=False):
    """Legacy manual exports omit polarity and require explicit selection."""
    with open(path, "rb") as stream:
        raw = stream.read(131073)
    if len(raw) > 131072:
        raise ValueError("参考文件过大")
    try:
        doc = json.loads(raw)
    except RecursionError as exc:
        raise ValueError("参考文件结构无效") from exc
    version = doc.get("version") if isinstance(doc, dict) else None
    if type(version) is not int or (version != 2 and not (allow_legacy and version == 1)):
        raise ValueError("参考文件版本不兼容")
    if doc.get("binding") != binding:
        raise ValueError("参考文件与当前传感器、协议或布局不匹配")
    if version == 2 and (direction not in ("both", "increase", "decrease")
                         or doc.get("direction") != direction):
        raise ValueError("参考文件与当前响应方向不匹配")
    return ReferenceProfile.from_dict(doc.get("reference"))

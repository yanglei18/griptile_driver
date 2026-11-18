import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from griptile_core.equalization import EqualizationProfile
from griptile_viewer.equalization_store import EqualizationSettings
from griptile_viewer.load_capture import LoadContext


@pytest.fixture
def context():
    return LoadContext("pad", "one", 1, "usb:id", "row20_col20_v3", "both", 3000.,
                       "pad", 1234, "baseline-and-threshold-digest", 1)


@pytest.fixture
def profile():
    return EqualizationProfile("both", (100., 200.) * 200, (300., 600.) * 200)


def test_trusted_binding_persists_same_reference_only(tmp_path, context, profile):
    model = SimpleNamespace(equalization_context=context)
    settings = EqualizationSettings(model, tmp_path)
    settings.adopt(profile, context)
    restored = EqualizationSettings(model, tmp_path)
    restored.sync()
    assert restored.active == profile
    restored.set_enabled(False)
    for changes in ({"hardware_id": "another device"}, {"sensor_id": "another"},
                    {"reference_digest": "another reference"}, {"layout_id": "another layout"},
                    {"direction": "decrease"}):
        model.equalization_context = replace(context, **changes)
        restored.sync()
        assert restored.profile is None
        with pytest.raises(ValueError):
            restored.adopt(profile, context)
    model.equalization_context = replace(context, session_id="two", stream_epoch=2, generation=7)
    restored.sync()
    assert restored.profile == profile and restored.active is None
    restored.set_enabled(True)
    assert restored.active == profile
    restored.clear()
    assert restored.profile is None and not list(tmp_path.iterdir())


def test_anonymous_device_stays_in_one_live_reference(tmp_path, context, profile):
    model = SimpleNamespace(equalization_context=replace(context, hardware_id=""))
    settings = EqualizationSettings(model, tmp_path)
    settings.adopt(profile, model.equalization_context)
    assert settings.active == profile and not settings.persistent
    assert not list(tmp_path.iterdir())
    model.equalization_context = replace(model.equalization_context, stream_epoch=2)
    settings.sync()
    assert settings.active is None
    settings.adopt(profile, model.equalization_context)
    model.equalization_context = None
    settings.sync()
    assert settings.profile is None
    assert not list(tmp_path.iterdir())


def test_saved_file_revalidated_and_atomic_failure_preserves_last_profile(tmp_path, context, profile, monkeypatch):
    model = SimpleNamespace(equalization_context=context)
    settings = EqualizationSettings(model, tmp_path)
    settings.adopt(profile, context)
    path, = tmp_path.iterdir()
    original = path.read_bytes()

    def cannot_write(*args):
        raise OSError("磁盘不可写")

    monkeypatch.setattr("griptile_viewer.equalization_store.save_json_atomic", cannot_write)
    with pytest.raises(OSError):
        settings.set_enabled(False)
    assert settings.active == profile and path.read_bytes() == original

    doc = json.loads(original)
    doc["profile"]["check_signed_adc"][4] *= 3
    for payload in (b"{}", b"not json", json.dumps(doc).encode(), b"x" * 131073):
        path.write_bytes(payload)
        restored = EqualizationSettings(model, tmp_path)
        restored.sync()
        assert restored.profile is None and restored.error
        assert path.read_bytes() == payload


def test_two_windows_reject_a_stale_overwrite(tmp_path, context, profile):
    model = SimpleNamespace(equalization_context=context)
    first, second = (EqualizationSettings(model, tmp_path) for _ in range(2))
    first.adopt(profile, context)
    second.sync()
    first.set_enabled(False)
    with pytest.raises(ValueError, match="另一窗口"):
        second.set_enabled(True)
    assert second.profile == profile and second.active is None

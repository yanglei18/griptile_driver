"""Hold Qt application and window owners through the final graphics events of each test."""

import pytest


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from griptile_viewer.window import pg
    return pg.mkQApp()


@pytest.fixture
def open_window(qt_app):
    from griptile_viewer.window import QtCore, ViewerWindow

    windows = []

    def create(*args, **kwargs):
        window = ViewerWindow(*args, **kwargs)
        windows.append(window)
        window.show()
        qt_app.processEvents()
        return window

    yield create
    # Keep Python owners alive while Qt discards queued events for their
    # graphics scenes and deletes all child widgets.
    for window in reversed(windows):
        window.close()
        window.deleteLater()
    qt_app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qt_app.processEvents()

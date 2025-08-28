"""
Disable all network access at runtime.

- On import, monkeypatch socket.connect, urllib.request.urlopen, and http.client to raise.
- Export init() and call it from CLI to reinforce the guard.
"""

from __future__ import annotations

import builtins
import socket


_original_socket = socket.socket


def _raise_network_disabled(*args, **kwargs):
    raise OSError("network disabled")


def _patch_socket():
    class NoNetSocket(_original_socket):  # type: ignore[misc]
        def connect(self, *args, **kwargs):  # type: ignore[override]
            raise OSError("network disabled")

        def connect_ex(self, *args, **kwargs):  # type: ignore[override]
            raise OSError("network disabled")

    socket.socket = NoNetSocket  # type: ignore[assignment]


def _patch_urllib_httpclient():
    try:
        import urllib.request  # noqa: F401
        urllib_request = builtins.__import__("urllib.request", fromlist=["*"])
        def _blocked(*args, **kwargs):
            raise OSError("network disabled")
        setattr(urllib_request, "urlopen", _blocked)
    except Exception:
        pass

    try:
        import http.client  # noqa: F401
        http_client = builtins.__import__("http.client", fromlist=["*"])
        class _NoNetHTTPConnection(http_client.HTTPConnection):  # type: ignore[attr-defined]
            def request(self, *args, **kwargs):  # type: ignore[override]
                raise OSError("network disabled")
        http_client.HTTPConnection = _NoNetHTTPConnection  # type: ignore[assignment]
        class _NoNetHTTPSConnection(http_client.HTTPSConnection):  # type: ignore[attr-defined]
            def request(self, *args, **kwargs):  # type: ignore[override]
                raise OSError("network disabled")
        http_client.HTTPSConnection = _NoNetHTTPSConnection  # type: ignore[assignment]
    except Exception:
        pass


def init() -> None:
    _patch_socket()
    _patch_urllib_httpclient()


# Apply on import
init()


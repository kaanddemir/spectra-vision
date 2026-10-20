"""Request checks for a server that only local, same-origin pages may use.

The Host check defeats DNS rebinding; the Origin and Fetch Metadata checks stop other sites open in
the same browser from reading or changing local data. Non-browser local clients send neither header.
"""
from __future__ import annotations

import json
from typing import Iterable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
)
_SAME_SITE_REQUESTS = ("same-origin", "none")


def _hostname(host: str) -> str:
    if host.startswith("["):
        return host[:host.find("]") + 1] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class LocalRequestGuard:
    """Reject requests addressed to a foreign host or sent by a foreign page; add security headers."""

    def __init__(self, app: ASGIApp, *, hostnames: Iterable[str], origins: Iterable[str]) -> None:
        self.app = app
        self.hostnames = frozenset(hostnames)
        self.origins = frozenset(origins)

    def refusal(self, scope: Scope) -> tuple[int, str] | None:
        headers = {}
        for key, value in scope["headers"]:
            headers.setdefault(key.decode("latin-1").lower(), value.decode("latin-1"))
        if _hostname(headers.get("host", "")).lower() not in self.hostnames:
            return 400, "invalid host header"
        origin = headers.get("origin")
        if origin is not None:
            return None if origin in self.origins else (403, "cross-origin request refused")
        site = headers.get("sec-fetch-site")
        if site is not None and site not in _SAME_SITE_REQUESTS:
            return 403, "cross-site request refused"
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                present = {key.lower() for key, _ in message.get("headers", [])}
                message["headers"] = list(message.get("headers", [])) + [
                    (key, value) for key, value in SECURITY_HEADERS if key not in present]
            await send(message)

        refused = self.refusal(scope)
        if refused is None:
            await self.app(scope, receive, send_with_headers)
            return
        status, detail = refused
        body = json.dumps({"detail": detail}).encode()
        await send_with_headers({"type": "http.response.start", "status": status, "headers": [
            (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

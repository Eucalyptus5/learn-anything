import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path

from aiohttp import web
from aiortc import RTCSessionDescription

from tutor.transport import Connection, negotiate

CLIENT_ROOT = Path(__file__).resolve().parent.parent / "client"
HOST = "127.0.0.1"
CONNECTIONS = web.AppKey("connections", set[Connection])

logger = logging.getLogger(__name__)


async def _close_connections(app: web.Application) -> None:
    await asyncio.gather(
        *(connection.close() for connection in app[CONNECTIONS]), return_exceptions=True
    )
    app[CONNECTIONS].clear()


def create_app(
    client_root: Path = CLIENT_ROOT,
    on_connection: Callable[[Connection], None] | None = None,
) -> web.Application:
    app = web.Application()
    app[CONNECTIONS] = set()

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(client_root / "index.html")

    async def script(request: web.Request) -> web.FileResponse:
        return web.FileResponse(client_root / "client.js")

    async def frame(request: web.Request) -> web.FileResponse:
        return web.FileResponse(client_root / "frame.html")

    async def visuals(request: web.Request) -> web.FileResponse:
        return web.FileResponse(client_root / "visuals.js")

    async def vendor(request: web.Request) -> web.FileResponse:
        return web.FileResponse(client_root / "vendor" / "mermaid.min.js")

    async def offer(request: web.Request) -> web.Response:
        try:
            payload = json.loads(await request.text())
        except json.JSONDecodeError:
            logger.warning("offer_rejected reason=not_json")
            return web.json_response({"error": "expected a json offer"}, status=400)
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("sdp"), str)
            or payload.get("type") != "offer"
        ):
            logger.warning("offer_rejected reason=malformed")
            return web.json_response({"error": "expected a json offer"}, status=400)

        try:
            connection, answer = await negotiate(
                RTCSessionDescription(sdp=payload["sdp"], type="offer")
            )
        except (AssertionError, ValueError):
            # aiortc's sdp parser asserts on a truncated m-line instead of raising ValueError
            logger.warning("offer_rejected reason=bad_sdp")
            return web.json_response({"error": "expected a json offer"}, status=400)
        request.app[CONNECTIONS].add(connection)

        def forget() -> None:
            request.app[CONNECTIONS].discard(connection)

        connection.on_close(forget)
        if on_connection is not None:
            on_connection(connection)
        logger.info("offer_answered live=%d", len(request.app[CONNECTIONS]))
        return web.json_response({"sdp": answer.sdp, "type": answer.type})

    app.router.add_get("/", index)
    app.router.add_get("/client.js", script)
    app.router.add_get("/frame.html", frame)
    app.router.add_get("/visuals.js", visuals)
    app.router.add_get("/vendor/mermaid.min.js", vendor)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(_close_connections)
    return app

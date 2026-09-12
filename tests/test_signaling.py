import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from yarl import URL

from tutor.signaling import CLIENT_ROOT, CONNECTIONS, HOST, SessionRequest, create_app
from tutor.transport import Connection

LOOPBACK_TIMEOUT_S = 20.0
INDEX = "<!doctype html><title>tutor</title>"
SCRIPT = "export const ready = true;\n"
FRAME = "<!doctype html><title>frame</title><div id=d></div>"
VISUALS = "export const visuals = true;\n"
VISUAL_CHECK = "<!doctype html><title>visual check</title><div id=canvas></div>"
BENCH_MERMAID = "<!doctype html><title>mermaid bench</title><div id=canvas></div>"
VENDOR = "globalThis.mermaid = {};\n"
JSON_HEADERS = {"Content-Type": "application/json"}
PLAIN_HEADERS = {"Content-Type": "text/plain;charset=UTF-8"}
SESSION = {"subject": "PPO", "folder": "", "starting_from": "I know policy gradients"}
FOREIGN_ORIGINS = {
    "null": lambda host: {"Origin": "null"},
    "foreign-host": lambda host: {"Origin": "http://evil.example"},
    "wrong-port": lambda host: {"Origin": str(host.with_port(host.port + 1))},
    "rebound-host": lambda host: {
        "Origin": f"http://evil.example:{host.port}",
        "Host": f"evil.example:{host.port}",
    },
}
MALFORMED = [
    "not json at all",
    json.dumps(["v=0", "offer"]),
    json.dumps({"type": "offer"}),
    json.dumps({"sdp": "v=0"}),
    json.dumps({"sdp": "v=0", "type": "answer"}),
    json.dumps({"sdp": 17, "type": "offer"}),
    json.dumps({"sdp": "v=0\r\nnonsense\r\n", "type": "offer"}),
    json.dumps({"sdp": "v=0\r\nm=audio\r\n", "type": "offer"}),
    json.dumps({"sdp": "v=0", "type": "offer"}),
    json.dumps({"sdp": "v=0", "type": "offer", "session": {"subject": ""}}),
    json.dumps({"sdp": "v=0", "type": "offer", "session": {"subject": "x", "extra": 1}}),
    json.dumps(
        {"sdp": "v=0", "type": "offer", "session": {"subject": "x", "folder": "/nonexistent/dir"}}
    ),
]


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[TestClient]:
    (tmp_path / "index.html").write_text(INDEX)
    (tmp_path / "client.js").write_text(SCRIPT)
    (tmp_path / "frame.html").write_text(FRAME)
    (tmp_path / "visuals.js").write_text(VISUALS)
    (tmp_path / "visual_check.html").write_text(VISUAL_CHECK)
    (tmp_path / "bench_mermaid.html").write_text(BENCH_MERMAID)
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "mermaid.min.js").write_text(VENDOR)
    test_client = TestClient(TestServer(create_app(tmp_path)))
    await test_client.start_server()
    yield test_client
    await test_client.close()


def offering_peer() -> RTCPeerConnection:
    return RTCPeerConnection(RTCConfiguration(iceServers=[]))


async def offer_body(pc: RTCPeerConnection, session: dict[str, object] | None = None) -> str:
    pc.createDataChannel("tutor")
    pc.addTrack(AudioStreamTrack())
    await pc.setLocalDescription(await pc.createOffer())
    body: dict[str, object] = {"sdp": pc.localDescription.sdp, "type": "offer"}
    body["session"] = SESSION if session is None else session
    return json.dumps(body)


async def offer(client: TestClient, pc: RTCPeerConnection) -> dict[str, object]:
    response = await client.post("/offer", data=await offer_body(pc), headers=JSON_HEADERS)
    assert response.status == 200
    return await response.json()


def test_the_client_root_is_the_repo_client_directory() -> None:
    assert CLIENT_ROOT == Path(__file__).resolve().parent.parent / "client"


def test_the_server_binds_loopback_and_not_every_interface() -> None:
    assert HOST == "127.0.0.1"
    assert HOST != "0.0.0.0"


async def test_the_browser_files_are_served_from_the_client_root(client: TestClient) -> None:
    index = await client.get("/")
    script = await client.get("/client.js")

    assert index.status == 200
    assert await index.text() == INDEX
    assert script.status == 200
    assert await script.text() == SCRIPT


async def test_the_frame_dispatcher_and_vendor_bundle_are_served(client: TestClient) -> None:
    frame = await client.get("/frame.html")
    visuals = await client.get("/visuals.js")
    vendor = await client.get("/vendor/mermaid.min.js")

    assert frame.status == 200
    assert await frame.text() == FRAME
    assert visuals.status == 200
    assert await visuals.text() == VISUALS
    assert vendor.status == 200
    assert await vendor.text() == VENDOR


async def test_the_render_frame_refuses_foreign_embedding(client: TestClient) -> None:
    frame = await client.get("/frame.html")

    assert frame.status == 200
    assert await frame.text() == FRAME
    assert frame.headers["Content-Security-Policy"] == "frame-ancestors 'self'"


async def test_the_host_page_carries_no_frame_ancestors_header(client: TestClient) -> None:
    index = await client.get("/")

    assert index.status == 200
    assert "Content-Security-Policy" not in index.headers


async def test_the_visual_check_harness_is_served(client: TestClient) -> None:
    harness = await client.get("/visual_check.html")

    assert harness.status == 200
    assert await harness.text() == VISUAL_CHECK


async def test_the_mermaid_benchmark_is_served(client: TestClient) -> None:
    harness = await client.get("/bench_mermaid.html")

    assert harness.status == 200
    assert await harness.text() == BENCH_MERMAID


async def test_a_missing_vendor_bundle_is_a_not_found_rather_than_a_crash(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text(INDEX)
    client = TestClient(TestServer(create_app(tmp_path)))
    await client.start_server()
    try:
        vendor = await client.get("/vendor/mermaid.min.js")
        index = await client.get("/")

        assert vendor.status == 404
        assert index.status == 200
    finally:
        await client.close()


async def test_an_offer_is_answered_with_a_gathered_answer(client: TestClient) -> None:
    pc = offering_peer()

    answer = await offer(client, pc)

    assert answer["type"] == "answer"
    assert "a=candidate" in answer["sdp"]
    await pc.close()


async def test_a_negotiated_connection_is_handed_to_the_callback(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(INDEX)
    (tmp_path / "client.js").write_text(SCRIPT)
    handed: list[Connection] = []
    client = TestClient(
        TestServer(create_app(tmp_path, on_connection=lambda c, r: handed.append(c)))
    )
    await client.start_server()
    pc = offering_peer()
    try:
        await offer(client, pc)

        assert handed == list(client.app[CONNECTIONS])
        assert len(handed) == 1
    finally:
        await pc.close()
        await client.close()


async def test_the_session_request_reaches_the_callback(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(INDEX)
    (tmp_path / "client.js").write_text(SCRIPT)
    seen: list[tuple[Connection, SessionRequest]] = []
    client = TestClient(
        TestServer(create_app(tmp_path, on_connection=lambda c, r: seen.append((c, r))))
    )
    await client.start_server()
    pc = offering_peer()
    try:
        session = {
            "subject": "PPO",
            "folder": str(tmp_path),
            "starting_from": "  I know policy gradients  ",
        }
        response = await client.post(
            "/offer", data=await offer_body(pc, session), headers=PLAIN_HEADERS
        )
        assert response.status == 200
        ((_, request),) = seen
        assert request.subject == "PPO"
        assert request.folder == tmp_path.resolve()
        assert request.starting_from == "I know policy gradients"
    finally:
        await pc.close()
        await client.close()


async def test_a_blank_folder_means_no_folder(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(INDEX)
    (tmp_path / "client.js").write_text(SCRIPT)
    seen: list[SessionRequest] = []
    client = TestClient(TestServer(create_app(tmp_path, on_connection=lambda c, r: seen.append(r))))
    await client.start_server()
    pc = offering_peer()
    try:
        response = await client.post("/offer", data=await offer_body(pc), headers=PLAIN_HEADERS)
        assert response.status == 200
        assert seen[0].folder is None
    finally:
        await pc.close()
        await client.close()


async def test_a_folder_that_is_a_file_is_rejected(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "paper.md").write_text("PPO")
    pc = offering_peer()
    session = {"subject": "PPO", "folder": str(tmp_path / "paper.md")}
    response = await client.post(
        "/offer", data=await offer_body(pc, session), headers=PLAIN_HEADERS
    )
    assert response.status == 400
    assert client.app[CONNECTIONS] == set()
    await pc.close()


async def test_a_folder_that_cannot_be_resolved_is_rejected(
    client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    pc = offering_peer()
    session = {"subject": "PPO", "folder": str(tmp_path / "a")}
    response = await client.post(
        "/offer", data=await offer_body(pc, session), headers=PLAIN_HEADERS
    )
    assert response.status == 400
    assert client.app[CONNECTIONS] == set()
    await pc.close()


@pytest.mark.parametrize("body", MALFORMED)
async def test_a_malformed_offer_is_rejected_without_a_connection(
    client: TestClient, body: str
) -> None:
    response = await client.post("/offer", data=body, headers=JSON_HEADERS)

    assert response.status == 400
    assert client.app[CONNECTIONS] == set()


@pytest.mark.parametrize("origin", FOREIGN_ORIGINS.values(), ids=FOREIGN_ORIGINS.keys())
async def test_an_offer_from_a_foreign_origin_is_rejected(
    tmp_path: Path, origin: Callable[[URL], dict[str, str]]
) -> None:
    (tmp_path / "index.html").write_text(INDEX)
    handed: list[Connection] = []
    client = TestClient(
        TestServer(create_app(tmp_path, on_connection=lambda c, r: handed.append(c)))
    )
    await client.start_server()
    pc = offering_peer()
    try:
        response = await client.post(
            "/offer",
            data=await offer_body(pc),
            headers={**PLAIN_HEADERS, **origin(client.make_url("/").origin())},
        )

        assert response.status == 403
        assert client.app[CONNECTIONS] == set()
        assert handed == []
    finally:
        await pc.close()
        await client.close()


async def test_a_foreign_origin_is_rejected_before_the_body_is_read(client: TestClient) -> None:
    response = await client.post(
        "/offer",
        data="not json at all",
        headers={**PLAIN_HEADERS, "Origin": "http://evil.example"},
    )

    assert response.status == 403
    assert client.app[CONNECTIONS] == set()


async def test_an_offer_from_the_host_origin_is_accepted(client: TestClient) -> None:
    pc = offering_peer()

    response = await client.post(
        "/offer",
        data=await offer_body(pc),
        headers={**PLAIN_HEADERS, "Origin": str(client.make_url("/").origin())},
    )

    assert response.status == 200
    assert (await response.json())["type"] == "answer"
    await pc.close()


async def test_an_offer_from_localhost_on_the_bound_port_is_accepted(client: TestClient) -> None:
    pc = offering_peer()

    response = await client.post(
        "/offer",
        data=await offer_body(pc),
        headers={**PLAIN_HEADERS, "Origin": f"http://localhost:{client.port}"},
    )

    assert response.status == 200
    assert (await response.json())["type"] == "answer"
    await pc.close()


async def test_shutdown_closes_the_survivors_of_an_already_closed_connection(
    client: TestClient,
) -> None:
    first = offering_peer()
    second = offering_peer()
    await offer(client, first)
    await offer(client, second)
    connections = list(client.app[CONNECTIONS])
    await connections[0].close()

    await client.close()

    assert len(connections) == 2
    for connection in connections:
        assert connection.closed
    await first.close()
    await second.close()


async def test_a_peer_that_goes_away_leaves_the_live_set(client: TestClient) -> None:
    pc = offering_peer()
    answer = await offer(client, pc)
    (connection,) = client.app[CONNECTIONS]
    gone = asyncio.Event()
    connection.on_close(gone.set)
    connected = asyncio.Event()

    @pc.on("connectionstatechange")
    def _on_state() -> None:
        if pc.connectionState == "connected":
            connected.set()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
    await asyncio.wait_for(connected.wait(), LOOPBACK_TIMEOUT_S)
    await pc.close()
    await asyncio.wait_for(gone.wait(), LOOPBACK_TIMEOUT_S)

    assert connection.closed
    assert client.app[CONNECTIONS] == set()

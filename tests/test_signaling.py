import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from aiortc import RTCConfiguration, RTCPeerConnection
from aiortc.mediastreams import AudioStreamTrack

from tutor.signaling import CLIENT_ROOT, CONNECTIONS, HOST, create_app

INDEX = "<!doctype html><title>tutor</title>"
SCRIPT = "export const ready = true;\n"
JSON_HEADERS = {"Content-Type": "application/json"}
MALFORMED = [
    "not json at all",
    json.dumps(["v=0", "offer"]),
    json.dumps({"type": "offer"}),
    json.dumps({"sdp": "v=0"}),
    json.dumps({"sdp": "v=0", "type": "answer"}),
    json.dumps({"sdp": 17, "type": "offer"}),
    json.dumps({"sdp": "v=0\r\nnonsense\r\n", "type": "offer"}),
    json.dumps({"sdp": "v=0\r\nm=audio\r\n", "type": "offer"}),
]


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[TestClient]:
    (tmp_path / "index.html").write_text(INDEX)
    (tmp_path / "client.js").write_text(SCRIPT)
    test_client = TestClient(TestServer(create_app(tmp_path)))
    await test_client.start_server()
    yield test_client
    await test_client.close()


def offering_peer() -> RTCPeerConnection:
    return RTCPeerConnection(RTCConfiguration(iceServers=[]))


async def offer(client: TestClient, pc: RTCPeerConnection) -> dict[str, object]:
    pc.createDataChannel("tutor")
    pc.addTrack(AudioStreamTrack())
    await pc.setLocalDescription(await pc.createOffer())
    response = await client.post(
        "/offer",
        data=json.dumps({"sdp": pc.localDescription.sdp, "type": "offer"}),
        headers=JSON_HEADERS,
    )
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


async def test_an_offer_is_answered_with_a_gathered_answer(client: TestClient) -> None:
    pc = offering_peer()

    answer = await offer(client, pc)

    assert answer["type"] == "answer"
    assert "a=candidate" in answer["sdp"]
    await pc.close()


@pytest.mark.parametrize("body", MALFORMED)
async def test_a_malformed_offer_is_rejected_without_a_connection(
    client: TestClient, body: str
) -> None:
    response = await client.post("/offer", data=body, headers=JSON_HEADERS)

    assert response.status == 400
    assert client.app[CONNECTIONS] == set()


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
        assert connection._pc.connectionState == "closed"
    await first.close()
    await second.close()

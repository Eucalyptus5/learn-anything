const connectButton = document.getElementById("connect");
const statusEl = document.getElementById("status");
const audioEl = document.getElementById("tutor");

const handlers = new Map();

const state = {
  stage: "idle",
  echoCancellation: "unknown",
  connection: "new",
  channel: "none",
  packetsSent: 0,
  packetsReceived: 0,
  audioLevel: 0,
};

let pc = null;
let channel = null;
let statsTimer = null;
let mic = null;

function render() {
  statusEl.textContent = [
    ["stage", state.stage],
    ["applied echoCancellation", state.echoCancellation],
    ["connection state", state.connection],
    ["data channel", state.channel],
    ["audio packets sent", state.packetsSent],
    ["audio packets received", state.packetsReceived],
    ["mic level", state.audioLevel.toFixed(4)],
  ]
    .map(([label, value]) => label.padEnd(24) + " " + value)
    .join("\n");
}

export function onJson(type, handler) {
  handlers.set(type, handler);
}

export function sendJson(obj) {
  channel.send(JSON.stringify(obj));
}

function release(stage) {
  state.stage = stage;
  clearInterval(statsTimer);
  statsTimer = null;
  if (mic !== null) {
    mic.getTracks().forEach((track) => track.stop());
    mic = null;
  }
  if (pc !== null) {
    const peer = pc;
    pc = null;
    peer.close();
    state.connection = "closed";
  }
  connectButton.disabled = false;
  render();
}

function gatheringComplete(peer) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    peer.addEventListener("icegatheringstatechange", () => {
      if (peer.iceGatheringState === "complete") resolve();
    });
  });
}

async function pollStats(peer) {
  const report = await peer.getStats();
  if (peer !== pc) return;
  report.forEach((entry) => {
    if (entry.kind !== "audio") return;
    if (entry.type === "outbound-rtp") state.packetsSent = entry.packetsSent;
    if (entry.type === "inbound-rtp") state.packetsReceived = entry.packetsReceived;
    if (entry.type === "media-source") state.audioLevel = entry.audioLevel ?? 0;
  });
  render();
}

async function connect() {
  state.stage = "requesting microphone";
  render();

  mic = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
  });
  const micTrack = mic.getAudioTracks()[0];
  const applied = micTrack.getSettings().echoCancellation;
  state.echoCancellation =
    applied === true ? "true" : String(applied) + " (requested true; the tutor will hear itself)";

  const peer = new RTCPeerConnection();
  pc = peer;

  peer.addEventListener("track", (event) => {
    audioEl.srcObject = new MediaStream([event.track]);
  });

  peer.addEventListener("connectionstatechange", () => {
    if (peer !== pc) return;
    state.connection = peer.connectionState;
    clearInterval(statsTimer);
    statsTimer = null;
    if (peer.connectionState === "connected") {
      statsTimer = setInterval(() => {
        pollStats(peer).catch((error) => release("stats unavailable: " + error));
      }, 1000);
    }
    if (["disconnected", "failed", "closed"].includes(peer.connectionState)) {
      release("connection " + peer.connectionState);
      return;
    }
    render();
  });

  channel = peer.createDataChannel("tutor");
  channel.addEventListener("open", () => {
    state.channel = "open";
    render();
  });
  channel.addEventListener("close", () => {
    state.channel = "closed";
    render();
  });
  channel.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    const handler = handlers.get(payload.type);
    if (handler) handler(payload);
  });
  state.channel = "connecting";

  peer.addTrack(micTrack, mic);
  await peer.setLocalDescription(await peer.createOffer());

  state.stage = "gathering ice candidates";
  render();
  await gatheringComplete(peer);

  state.stage = "posting offer";
  render();
  const response = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: peer.localDescription.sdp, type: "offer" }),
  });
  if (!response.ok) {
    release("offer rejected with " + response.status);
    return;
  }

  await peer.setRemoteDescription(new RTCSessionDescription(await response.json()));
  state.stage = "answer applied";
  render();
}

connectButton.onclick = () => {
  connectButton.disabled = true;
  connect().catch((error) => release("failed: " + error));
};

render();

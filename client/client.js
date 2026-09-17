import { entries, mount, onPayload, receive, reset, show, theme } from "/visuals.js";

const app = document.querySelector(".app");
const welcome = document.querySelector(".welcome");
const composer = document.querySelector(".composer");
const subjectInput = composer.querySelector(".subject");
const folderInput = composer.querySelector(".folder");
const startingInput = composer.querySelector(".starting");
const note = composer.querySelector(".note");
const connectButton = composer.querySelector(".connect");
const bar = document.querySelector(".bar");
const heading = bar.querySelector("h2");
const phaseChip = bar.querySelector(".chip.phase");
const phaseText = phaseChip.lastElementChild;
const liveText = bar.querySelector(".chip.live").lastElementChild;
const debug = document.querySelector(".debug");
const stage = document.querySelector(".stage");
const canvasTitle = stage.querySelector(".canvas .title");
const caption = stage.querySelector(".caption");
const say = document.getElementById("say");
const prev = caption.querySelector(".prev");
const cur = caption.querySelector(".cur");
const empty = document.querySelector(".side .empty");
const historyList = document.querySelector(".history");
const sessionLine = document.querySelector(".session");
const sessionSubject = sessionLine.querySelector("b");
const sessionFolder = sessionLine.querySelector("span");
const drawer = document.querySelector(".drawer");
const thread = drawer.querySelector(".thread");
const audioEl = document.getElementById("tutor");
const dark = matchMedia("(prefers-color-scheme: dark)");

const handlers = new Map();
const openHandlers = [];
const turns = [];

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
let card = null;

function render() {
  debug.textContent = [
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

export function onOpen(handler) {
  openHandlers.push(handler);
}

export function sendJson(obj) {
  channel.send(JSON.stringify(obj));
}

function release(reason) {
  state.stage = reason;
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
  note.textContent = reason;
  welcome.hidden = false;
  bar.hidden = true;
  stage.hidden = true;
  connectButton.disabled = false;
  render();
}

function begin() {
  heading.textContent = card.subject;
  sessionSubject.textContent = card.subject;
  sessionFolder.textContent = card.folder || "No folder";
  sessionLine.hidden = false;
  prev.textContent = "";
  cur.textContent = "";
  say.value = "";
  caption.classList.remove("speaking");
  thread.replaceChildren();
  turns.length = 0;
  liveText.textContent = "listening";
  phaseText.textContent = "teach";
  phaseChip.dataset.phase = "teach";
  welcome.hidden = true;
  bar.hidden = false;
  stage.hidden = false;
}

function sendTheme() {
  const name = dark.matches ? "dark" : "light";
  sendJson({ type: "theme", theme: name });
  theme(name);
}

function download(name, type, text) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function markdown() {
  const lines = [`# ${card.subject}`, ""];
  turns.forEach((turn, i) => {
    lines.push(`## turn ${i + 1} (${turn.phase})`, "");
    lines.push(`**you:** ${turn.learner}`, "");
    lines.push(`**tutor:** ${turn.tutor.join(" ")}`, "");
  });
  lines.push("## visuals", "");
  for (const { title } of entries()) lines.push(`_visual: ${title}_`);
  return lines.join("\n") + "\n";
}

function exportSession(format) {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const stamp =
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `-${pad(now.getHours())}${pad(now.getMinutes())}`;
  const slug = card.subject.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  const name = `tutor-${slug}-${stamp}`;
  if (format === "json") {
    const session = {
      version: 1,
      subject: card.subject,
      starting_from: card.starting_from,
      theme: dark.matches ? "dark" : "light",
      exported_at: now.toISOString(),
      turns,
      visuals: entries(),
    };
    download(`${name}.json`, "application/json", JSON.stringify(session, null, 2));
  } else {
    download(`${name}.md`, "text/markdown", markdown());
  }
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
    openHandlers.forEach((handler) => handler());
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
    body: JSON.stringify({
      sdp: peer.localDescription.sdp,
      type: "offer",
      session: {
        subject: card.subject,
        folder: card.folder,
        starting_from: card.starting_from,
      },
    }),
  });
  if (!response.ok) {
    release("offer rejected with " + response.status);
    return;
  }

  await peer.setRemoteDescription(new RTCSessionDescription(await response.json()));
  state.stage = "answer applied";
  render();
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  card = {
    subject: subjectInput.value,
    folder: folderInput.value,
    starting_from: startingInput.value,
  };
  connectButton.disabled = true;
  connect().catch((error) => release("failed: " + error));
});

document.addEventListener("keydown", (event) => {
  if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  const open = channel !== null && channel.readyState === "open";
  if (event.key === "t") {
    drawer.hidden = !drawer.hidden;
    app.classList.toggle("with-drawer", !drawer.hidden);
  } else if (event.key === "d") {
    debug.hidden = !debug.hidden;
  } else if (event.key === "e" && open) {
    exportSession("json");
  } else if (event.key === "m" && open) {
    exportSession("markdown");
  }
});

say.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.isComposing) {
    event.preventDefault();
    const text = say.value.trim();
    if (text !== "" && channel !== null && channel.readyState === "open") {
      sendJson({ type: "say", text });
      say.value = "";
    }
  }
});

dark.addEventListener("change", () => {
  if (channel !== null && channel.readyState === "open") sendTheme();
});

onPayload("state", (payload) => {
  liveText.textContent = payload.state;
  phaseText.textContent = payload.phase;
  phaseChip.dataset.phase = payload.phase;
  caption.classList.toggle("speaking", payload.state === "speaking");
  if (payload.state === "thinking") turns[turns.length - 1].phase = payload.phase;
});

onPayload("caption", (payload) => {
  turns.find((turn) => turn.turn_id === payload.turn_id).tutor.push(payload.text);
  prev.textContent = cur.textContent;
  cur.textContent = payload.text;
  const last = thread.lastElementChild;
  if (last !== null && last.classList.contains("tutor") && last.dataset.turn === payload.turn_id) {
    last.textContent += " " + payload.text;
  } else {
    const line = document.createElement("p");
    line.className = "tutor";
    line.dataset.turn = payload.turn_id;
    line.textContent = payload.text;
    thread.append(line);
  }
  thread.scrollTop = thread.scrollHeight;
});

onPayload("transcript", (payload) => {
  turns.push({ turn_id: payload.turn_id, phase: null, learner: payload.text, tutor: [] });
  const line = document.createElement("p");
  line.className = "learner";
  line.textContent = payload.text;
  thread.append(line);
  thread.scrollTop = thread.scrollHeight;
});

onPayload("history", (items, current) => {
  historyList.replaceChildren(
    ...items.map(({ i, title }) => {
      const button = document.createElement("button");
      button.textContent = title;
      if (i === current) button.setAttribute("aria-current", "true");
      button.addEventListener("click", () => show(i));
      return button;
    }),
  );
  empty.hidden = items.length > 0;
  canvasTitle.textContent = current >= 0 ? items[current].title : "";
});

mount(stage.querySelector(".canvas"));
onJson("diagram.push", receive);
onJson("diagram.clear", receive);
onJson("source.highlight", receive);
onJson("app.push", receive);
onJson("state", receive);
onJson("caption", receive);
onJson("transcript", receive);
onOpen(reset);
onOpen(begin);
onOpen(sendTheme);
render();

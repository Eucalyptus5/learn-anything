const KINDS = new Set(["flowchart", "sequence"]);
const STATES = new Set(["listening", "thinking", "speaking"]);
const PHASES = new Set(["teach", "concrete", "interrogate"]);
const KEYS = {
  "diagram.push": ["type", "seq", "id", "kind", "source", "title"],
  "diagram.clear": ["type", "seq"],
  "source.highlight": ["type", "seq", "path", "start_line", "end_line"],
  "app.push": ["type", "seq", "id", "html", "title"],
  "state": ["type", "seq", "state", "phase"],
  "caption": ["type", "seq", "turn_id", "text"],
  "transcript": ["type", "seq", "turn_id", "text"],
};
const CAPS = { id: 64, source: 8000, html: 64000, path: 4096, title: 80, turn_id: 32 };
const TEXT_CAPS = { caption: 2000, transcript: 4000 };
const listeners = new Map();
const history = [];

let canvas = null;
let frame = null;
let highlight = null;
let app = null;
let loaded = Promise.resolve();
let lastSeq = 0;
let themeSeq = 0;

function reject(reason) {
  console.warn("visual payload rejected: " + reason);
  return false;
}

function cappedString(payload, key, cap = CAPS[key]) {
  const value = payload[key];
  return typeof value === "string" && [...value].length <= cap;
}

function positiveInteger(value) {
  return Number.isInteger(value) && value >= 1;
}

export function validate(payload) {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    return reject("not an object");
  }
  if (!Object.hasOwn(KEYS, payload.type)) return reject("unknown type");
  const expected = KEYS[payload.type];
  const keys = Object.keys(payload);
  if (keys.length !== expected.length || !expected.every((key) => keys.includes(key))) {
    return reject("unexpected keys for " + payload.type);
  }
  if (!positiveInteger(payload.seq)) return reject("seq is not a positive integer");
  switch (payload.type) {
    case "diagram.push":
      if (!cappedString(payload, "id")) return reject("bad id");
      if (!KINDS.has(payload.kind)) return reject("unknown kind");
      if (!cappedString(payload, "source")) return reject("bad source");
      if (!cappedString(payload, "title")) return reject("bad title");
      return true;
    case "diagram.clear":
      return true;
    case "source.highlight":
      if (!cappedString(payload, "path")) return reject("bad path");
      if (!positiveInteger(payload.start_line)) return reject("bad start_line");
      if (!positiveInteger(payload.end_line)) return reject("bad end_line");
      if (payload.end_line < payload.start_line) return reject("end_line precedes start_line");
      return true;
    case "app.push":
      if (!cappedString(payload, "id")) return reject("bad id");
      if (!cappedString(payload, "html")) return reject("bad html");
      if (!cappedString(payload, "title")) return reject("bad title");
      return true;
    case "state":
      if (!STATES.has(payload.state)) return reject("unknown state");
      if (!PHASES.has(payload.phase)) return reject("unknown phase");
      return true;
    case "caption":
    case "transcript":
      if (!cappedString(payload, "turn_id")) return reject("bad turn_id");
      if (!cappedString(payload, "text", TEXT_CAPS[payload.type])) return reject("bad text");
      return true;
  }
}

export function onPayload(type, handler) {
  listeners.set(type, handler);
}

function sandboxedFrame() {
  const element = document.createElement("iframe");
  element.setAttribute("sandbox", "allow-scripts");
  element.setAttribute("allow", "");
  element.setAttribute("referrerpolicy", "no-referrer");
  return element;
}

export function mount(root) {
  canvas = root;
  frame = sandboxedFrame();
  loaded = new Promise((resolve) => frame.addEventListener("load", resolve, { once: true }));
  frame.src = "/frame.html";
  highlight = document.createElement("div");
  highlight.className = "highlight";
  canvas.append(frame, highlight);
}

function post(message) {
  loaded = loaded.then(() => frame.contentWindow.postMessage(message, "*"));
}

function unmountApp() {
  if (app !== null) app.remove();
  app = null;
  frame.hidden = false;
}

function render(payload) {
  unmountApp();
  if (payload.type === "diagram.push") {
    post({ seq: payload.seq, kind: payload.kind, source: payload.source });
    return;
  }
  app = sandboxedFrame();
  app.srcdoc = payload.html;
  canvas.append(app);
  frame.hidden = true;
}

function announce(current) {
  listeners.get("history")?.(history.map((h, i) => ({ i, title: h.title })), current);
}

export function receive(payload) {
  if (!validate(payload)) return;
  if (payload.seq <= lastSeq) return;
  lastSeq = payload.seq;
  switch (payload.type) {
    case "diagram.push":
    case "app.push":
      history.push({ payload, title: payload.title });
      render(payload);
      announce(history.length - 1);
      break;
    case "diagram.clear":
      unmountApp();
      post({ seq: payload.seq, clear: true });
      announce(-1);
      break;
    case "source.highlight":
      highlight.textContent =
        payload.path + ":" + payload.start_line + "-" + payload.end_line;
      break;
    case "state":
    case "caption":
    case "transcript":
      listeners.get(payload.type)?.(payload);
      break;
  }
}

export function show(i) {
  render(history[i].payload);
  announce(i);
}

export function theme(name) {
  post({ seq: ++themeSeq, theme: name });
}

export function reset() {
  lastSeq = 0;
  history.length = 0;
  unmountApp();
  highlight.textContent = "";
  post({ seq: 0, clear: true });
  announce(-1);
}

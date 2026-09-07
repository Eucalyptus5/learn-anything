const KINDS = new Set(["flowchart", "sequence"]);
const KEYS = {
  "diagram.push": ["type", "seq", "id", "kind", "source"],
  "diagram.clear": ["type", "seq"],
  "source.highlight": ["type", "seq", "path", "start_line", "end_line"],
  "app.push": ["type", "seq", "id", "html"],
};
const CAPS = { id: 64, source: 8000, html: 64000, path: 4096 };

let frame = null;
let highlight = null;
let loaded = Promise.resolve();
let lastSeq = 0;

function reject(reason) {
  console.warn("visual payload rejected: " + reason);
  return false;
}

function cappedString(payload, key) {
  const value = payload[key];
  return typeof value === "string" && [...value].length <= CAPS[key];
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
      return true;
  }
}

export function mount(canvas) {
  frame = document.createElement("iframe");
  frame.setAttribute("sandbox", "allow-scripts");
  frame.setAttribute("allow", "");
  frame.setAttribute("referrerpolicy", "no-referrer");
  loaded = new Promise((resolve) => frame.addEventListener("load", resolve, { once: true }));
  frame.src = "/frame.html";
  highlight = document.createElement("div");
  canvas.append(frame, highlight);
}

function post(message) {
  loaded = loaded.then(() => frame.contentWindow.postMessage(message, "*"));
}

export function receive(payload) {
  if (!validate(payload)) return;
  if (payload.seq <= lastSeq) return;
  lastSeq = payload.seq;
  switch (payload.type) {
    case "diagram.push":
      post({ seq: payload.seq, kind: payload.kind, source: payload.source });
      break;
    case "diagram.clear":
      post({ seq: payload.seq, clear: true });
      break;
    case "source.highlight":
      highlight.textContent =
        payload.path + ":" + payload.start_line + "-" + payload.end_line;
      break;
  }
}

export function reset() {
  lastSeq = 0;
}

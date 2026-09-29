const KINDS = new Set(["flowchart", "sequence"]);
const STATES = new Set(["listening", "thinking", "speaking"]);
const SCENE_ID = /^[a-z][a-z0-9-]{0,31}$/;
const STATUSES = new Set(["planned", "building", "built", "failed", "done"]);
const KEYS = {
  "diagram.push": ["type", "seq", "id", "kind", "source", "title"],
  "diagram.clear": ["type", "seq"],
  "source.highlight": ["type", "seq", "path", "start_line", "end_line"],
  "app.push": ["type", "seq", "id", "html", "title"],
  "state": ["type", "seq", "state", "interrupted"],
  "caption": ["type", "seq", "turn_id", "text", "lead_ms"],
  "transcript": ["type", "seq", "turn_id", "text"],
  "scene.push": ["type", "seq", "scene_id", "title", "html", "steps"],
  "scene.show": ["type", "seq", "scene_id", "at"],
  "scene.step": ["type", "seq", "scene_id", "n", "lead_ms"],
  "lesson.attach": ["type", "seq", "epoch"],
  "lesson.cue": ["type", "seq", "epoch", "barrier", "cue_id", "chunk_id", "scene_id", "revision", "lead_ms", "audio_ms", "tag"],
  "lesson.sync": ["type", "seq", "epoch", "barrier"],
  "lesson.state": ["type", "seq", "scenes", "current"],
};
const CAPS = { id: 64, source: 8000, html: 64000, path: 4096, title: 80, turn_id: 32 };
const TEXT_CAPS = { caption: 2000, transcript: 4000 };
const SCENE_HTML_CAP = 200000;
const STEP_CAP = 120;
const STEPS_MAX = 8;
const STEP_MAX = 5;
const SCENES_MAX = 12;
const ERROR_CAP = 500;
const FADE_MS = 450;
const listeners = new Map();
const history = [];

let canvas = null;
let frame = null;
let highlight = null;
let app = null;
let checking = null;
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

function sceneId(value) {
  return typeof value === "string" && SCENE_ID.test(value);
}

function stepList(value) {
  return (
    Array.isArray(value) &&
    value.length >= 1 &&
    value.length <= STEPS_MAX &&
    value.every((say) => typeof say === "string" && say !== "" && [...say].length <= STEP_CAP)
  );
}

function nonNegative(value) {
  return Number.isInteger(value) && value >= 0;
}

function exactKeys(value, expected) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const keys = Object.keys(value);
  return keys.length === expected.length && expected.every((key) => keys.includes(key));
}

function cueTag(tag) {
  if (exactKeys(tag, ["kind", "n"]) && tag.kind === "step") {
    return Number.isInteger(tag.n) && tag.n >= 1 && tag.n <= STEP_MAX;
  }
  if (exactKeys(tag, ["kind", "n", "scene_id"]) && tag.kind === "scene") {
    return Number.isInteger(tag.n) && tag.n >= 1 && tag.n <= SCENES_MAX && sceneId(tag.scene_id);
  }
  return false;
}

function sceneRow(row) {
  return (
    exactKeys(row, ["id", "title", "status"]) &&
    sceneId(row.id) &&
    cappedString(row, "title") &&
    STATUSES.has(row.status)
  );
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
      if (typeof payload.interrupted !== "boolean") return reject("bad interrupted");
      return true;
    case "caption":
      if (!cappedString(payload, "turn_id")) return reject("bad turn_id");
      if (!cappedString(payload, "text", TEXT_CAPS.caption)) return reject("bad text");
      if (!Number.isInteger(payload.lead_ms) || payload.lead_ms < 0) return reject("bad lead_ms");
      return true;
    case "transcript":
      if (!cappedString(payload, "turn_id")) return reject("bad turn_id");
      if (!cappedString(payload, "text", TEXT_CAPS.transcript)) return reject("bad text");
      return true;
    case "scene.push":
      if (!sceneId(payload.scene_id)) return reject("bad scene_id");
      if (!cappedString(payload, "title")) return reject("bad title");
      if (!cappedString(payload, "html", SCENE_HTML_CAP) || payload.html === "") {
        return reject("bad html");
      }
      if (!stepList(payload.steps)) return reject("bad steps");
      return true;
    case "scene.show":
      if (!sceneId(payload.scene_id)) return reject("bad scene_id");
      if (!positiveInteger(payload.at)) return reject("bad at");
      return true;
    case "scene.step":
      if (!sceneId(payload.scene_id)) return reject("bad scene_id");
      if (!positiveInteger(payload.n)) return reject("bad n");
      if (!Number.isInteger(payload.lead_ms) || payload.lead_ms < 0) return reject("bad lead_ms");
      return true;
    case "lesson.attach":
      if (!positiveInteger(payload.epoch)) return reject("bad epoch");
      return true;
    case "lesson.cue":
      if (!positiveInteger(payload.epoch)) return reject("bad epoch");
      if (!nonNegative(payload.barrier)) return reject("bad barrier");
      if (!positiveInteger(payload.cue_id)) return reject("bad cue_id");
      if (!nonNegative(payload.chunk_id)) return reject("bad chunk_id");
      if (payload.scene_id !== null && !sceneId(payload.scene_id)) return reject("bad scene_id");
      if (!nonNegative(payload.revision)) return reject("bad revision");
      if (!nonNegative(payload.lead_ms)) return reject("bad lead_ms");
      if (!nonNegative(payload.audio_ms)) return reject("bad audio_ms");
      if (!cueTag(payload.tag)) return reject("bad tag");
      return true;
    case "lesson.sync":
      if (!positiveInteger(payload.epoch)) return reject("bad epoch");
      if (!positiveInteger(payload.barrier)) return reject("bad barrier");
      return true;
    case "lesson.state":
      if (!Array.isArray(payload.scenes) || payload.scenes.length > SCENES_MAX) {
        return reject("bad scenes");
      }
      if (!payload.scenes.every(sceneRow)) return reject("bad scene row");
      if (payload.current !== null && !sceneId(payload.current)) return reject("bad current");
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
  element.classList.add("landing");
  element.ready = new Promise((resolve) => element.addEventListener("load", resolve, { once: true }));
  element.addEventListener("load", () => element.classList.remove("landing"), { once: true });
  return element;
}

export function mount(root) {
  canvas = root;
  frame = sandboxedFrame();
  loaded = frame.ready;
  frame.src = "/frame.html";
  highlight = document.createElement("div");
  highlight.className = "highlight";
  canvas.append(frame, highlight);
}

function post(message) {
  loaded = loaded.then(() => frame.contentWindow.postMessage(message, "*"));
}

function dropChecking() {
  if (checking !== null) checking.frame.remove();
  checking = null;
}

function unmountApp() {
  app = null;
  for (const gone of canvas.querySelectorAll("iframe")) {
    if (gone !== frame && (checking === null || gone !== checking.frame)) gone.remove();
  }
  frame.hidden = false;
}

function mountApp(element) {
  const previous = app;
  app = element;
  frame.hidden = true;
  if (previous === null) return;
  element.ready.then(() => {
    previous.classList.add("leaving");
    setTimeout(() => previous.remove(), FADE_MS);
  });
}

function render(payload) {
  if (payload.type === "diagram.push") {
    unmountApp();
    post({ seq: payload.seq, kind: payload.kind, source: payload.source });
    return;
  }
  const element = sandboxedFrame();
  element.srcdoc = payload.html;
  canvas.append(element);
  mountApp(element);
}

function check(payload) {
  dropChecking();
  const element = sandboxedFrame();
  element.classList.add("checking");
  element.srcdoc = payload.html;
  canvas.append(element);
  checking = { payload, frame: element };
}

function promote(at) {
  const { payload, frame: element } = checking;
  checking = null;
  element.classList.add("landing");
  element.classList.remove("checking");
  requestAnimationFrame(() => element.classList.remove("landing"));
  mountApp(element);
  history.push({ payload, title: payload.title });
  announce(history.length - 1);
  if (at > 1) stepScene(at);
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
    case "scene.push":
      check(payload);
      break;
    case "scene.show":
      if (checking === null || checking.payload.scene_id !== payload.scene_id) {
        reject("no checked scene " + payload.scene_id);
        break;
      }
      promote(payload.at);
      break;
    case "scene.step":
      listeners.get("step")?.(payload);
      break;
    case "lesson.attach":
      listeners.get("attach")?.(payload);
      break;
    case "lesson.cue":
      listeners.get("cue")?.(payload);
      break;
    case "lesson.sync":
      listeners.get("sync")?.(payload);
      break;
    case "lesson.state":
      listeners.get("lesson")?.(payload);
      break;
    case "state":
    case "caption":
    case "transcript":
      listeners.get(payload.type)?.(payload);
      break;
  }
}

export function stepScene(n) {
  if (app === null) return;
  const target = app;
  target.ready.then(() => {
    if (target === app) target.contentWindow.postMessage({ step: n }, "*");
  });
}

export function show(i) {
  render(history[i].payload);
  announce(i);
}

export function entries() {
  return history.map((h) => ({ title: h.title, payload: h.payload }));
}

export function blank() {
  unmountApp();
  post({ seq: 0, clear: true });
}

export function land(sceneId, at) {
  if (checking === null || !checking.reported || checking.payload.scene_id !== sceneId) {
    return false;
  }
  promote(at);
  return true;
}

export function theme(name) {
  post({ seq: ++themeSeq, theme: name });
}

export function reset() {
  lastSeq = 0;
  history.length = 0;
  dropChecking();
  unmountApp();
  highlight.textContent = "";
  post({ seq: 0, clear: true });
  announce(-1);
}

window.addEventListener("message", (event) => {
  if (checking === null || event.source !== checking.frame.contentWindow) return;
  const m = event.data;
  if (typeof m !== "object" || m === null || m.type !== "scene.ready") return;
  if (!Number.isInteger(m.steps) || !Number.isFinite(m.width) || !Number.isFinite(m.height)) return;
  if (typeof m.error !== "string") return;
  if (checking.reported) return;
  checking.reported = true;
  const { payload } = checking;
  let error = m.error.slice(0, ERROR_CAP);
  const counted = m.steps === payload.steps.length;
  const sized = m.width > 0 && m.height > 0;
  if (error === "" && !counted) error = "reported " + m.steps + " steps, pushed " + payload.steps.length;
  if (error === "" && !sized) error = "root has no size " + Math.round(m.width) + "x" + Math.round(m.height);
  const ok = error === "";
  const steps = Math.min(Math.max(m.steps, 0), STEPS_MAX);
  if (!ok) dropChecking();
  listeners.get("ready")?.({ scene_id: payload.scene_id, ok, steps, error });
});

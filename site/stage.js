import { FADE, exchange } from "./exchange.js";
import sine from "./lessons/sine.js";
import light from "./lessons/light.js";
import route from "./lessons/route.js";
import moon from "./lessons/moon.js";

const LESSONS = [sine, light, route, moon];
const WAIT = 0.6;
const TAIL = 0.25;
const RAMP = 0.25;

const main = document.querySelector("main");
const svg = document.querySelector(".drawing");
const sentence = document.querySelector(".sentence");
const you = document.querySelector(".you");
const tutor = document.querySelector(".tutor");
const question = you.querySelector(".asked");
const reduce = matchMedia("(prefers-reduced-motion: reduce)");

let mode = reduce.matches ? "reduce" : "play";
let current = 0;
let s = -WAIT;
let heard = [];
let spans = [];
let prev = null;

const scenes = LESSONS.map((lesson) => {
  const pose = lesson.build(svg);
  const group = svg.lastElementChild;
  group.style.display = "none";
  const timeline = exchange(lesson);
  const answer = lesson.words.map(([word, at]) => [word, timeline.pre + at]);
  return { lesson, pose, group, answer, ...timeline };
});

const marks = LESSONS.map((lesson, index) => {
  const mark = document.createElement("button");
  mark.type = "button";
  mark.className = "mark";
  mark.setAttribute("aria-label", lesson.name);
  mark.addEventListener("click", () => play(index, mode === "play" ? 0 : finished(index)));
  return mark;
});
document.querySelector(".marks").append(...marks);

const list = document.createElement("ol");
list.className = "sr-only";
for (const lesson of LESSONS) {
  const item = document.createElement("li");
  const answer = lesson.words.map(([word]) => word).join(" ");
  item.textContent = `You: ${lesson.asked} Tutor: ${answer}`;
  list.append(item);
}
sentence.after(list);

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function finished(index) {
  return scenes[index].total - FADE;
}

function fill(element, words) {
  const made = words.map(([word]) => {
    const span = document.createElement("span");
    span.textContent = word;
    return span;
  });
  element.replaceChildren(...made.flatMap((span, i) => (i === 0 ? [span] : [" ", span])));
  return made;
}

function reveal(made, words) {
  made.forEach((span, i) => span.classList.toggle("said", s > words[i][1]));
}

function talking(words) {
  return s > words[0][1] && s < words.at(-1)[1] + TAIL;
}

function play(index, at) {
  current = index;
  s = at;
  heard = fill(question, scenes[index].asked);
  main.classList.toggle("still", mode !== "play");
  scenes.forEach((scene, i) => {
    scene.group.style.display = i === index ? "" : "none";
  });
  for (const mark of marks) mark.style.setProperty("--p", "0");
  spans = fill(sentence, scenes[index].answer);
  // A span inserted and marked said in the same frame skips its fade; settle it unsaid first.
  void sentence.offsetWidth;
  show();
}

function show() {
  const { lesson, pose, group, pre, asked, answer, total } = scenes[current];
  pose(clamp(s - pre, 0, lesson.length));
  reveal(heard, asked);
  reveal(spans, answer);
  const fade = clamp((total - s) / FADE, 0, 1);
  group.style.opacity = String(fade);
  sentence.style.opacity = String(fade);
  you.style.opacity = String(Math.min(fade, clamp(s / RAMP, 0, 1)));
  tutor.style.opacity = String(Math.min(fade, clamp((s - pre) / RAMP, 0, 1)));
  you.classList.toggle("talking", talking(asked));
  tutor.classList.toggle("talking", talking(answer));
  marks[current].style.setProperty("--p", String(mode === "reduce" ? 1 : s / total));
}

function tick(dt) {
  if (s < 0) {
    s += dt;
    if (s >= 0) play(0, 0);
    return;
  }
  s += dt;
  if (s < scenes[current].total) show();
  else play((current + 1) % LESSONS.length, 0);
}

function frame(now) {
  if (mode === "play" && prev !== null && !document.hidden) tick((now - prev) / 1000);
  prev = now;
  requestAnimationFrame(frame);
}

document.addEventListener("visibilitychange", () => {
  prev = null;
});

reduce.addEventListener("change", () => {
  if (mode === "freeze") return;
  mode = reduce.matches ? "reduce" : "play";
  play(current, mode === "reduce" ? finished(current) : 0);
});

const freeze = /^#(\d+)@(\d*\.?\d+)$/.exec(location.hash);
const frozen = freeze === null ? -1 : Number(freeze[1]) - 1;
if (frozen >= 0 && frozen < LESSONS.length && Number(freeze[2]) <= scenes[frozen].total) {
  mode = "freeze";
  play(frozen, Number(freeze[2]));
} else if (mode === "reduce") {
  play(0, finished(0));
}
requestAnimationFrame(frame);

import sine from "./lessons/sine.js";
import light from "./lessons/light.js";
import route from "./lessons/route.js";
import moon from "./lessons/moon.js";

const LESSONS = [sine, light, route, moon];
const WAIT = 0.6;
const FADE = 0.8;

const svg = document.querySelector(".drawing");
const sentence = document.querySelector(".sentence");
const reduce = matchMedia("(prefers-reduced-motion: reduce)");

let mode = reduce.matches ? "reduce" : "play";
let current = 0;
let t = -WAIT;
let spans = [];
let prev = null;

const scenes = LESSONS.map((lesson) => {
  const pose = lesson.build(svg);
  const group = svg.lastElementChild;
  group.style.display = "none";
  return { lesson, pose, group };
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
  item.textContent = lesson.words.map(([word]) => word).join(" ");
  list.append(item);
}
sentence.after(list);

function finished(index) {
  return LESSONS[index].length - FADE;
}

function play(index, at) {
  current = index;
  t = at;
  spans = LESSONS[index].words.map(([word]) => {
    const span = document.createElement("span");
    span.textContent = word;
    return span;
  });
  sentence.classList.toggle("still", mode !== "play");
  sentence.replaceChildren(...spans.flatMap((span, i) => (i === 0 ? [span] : [" ", span])));
  scenes.forEach((scene, i) => {
    scene.group.style.display = i === index ? "" : "none";
  });
  for (const mark of marks) mark.style.setProperty("--p", "0");
  show();
}

function show() {
  const { lesson, pose, group } = scenes[current];
  pose(t);
  spans.forEach((span, i) => span.classList.toggle("said", t > lesson.words[i][1]));
  const fade = String(Math.min(1, Math.max(0, (lesson.length - t) / FADE)));
  group.style.opacity = fade;
  sentence.style.opacity = fade;
  marks[current].style.setProperty("--p", String(mode === "reduce" ? 1 : t / lesson.length));
}

function tick(dt) {
  if (t < 0) {
    t += dt;
    if (t >= 0) play(0, 0);
    return;
  }
  t += dt;
  if (t < LESSONS[current].length) show();
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
if (frozen >= 0 && frozen < LESSONS.length && Number(freeze[2]) <= LESSONS[frozen].length) {
  mode = "freeze";
  play(frozen, Number(freeze[2]));
} else if (mode === "reduce") {
  play(0, finished(0));
}
requestAnimationFrame(frame);

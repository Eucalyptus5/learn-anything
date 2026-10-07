import assert from "node:assert/strict";
import { test } from "node:test";

import { exchange } from "../../site/exchange.js";
import light, {
  INCIDENCE,
  N,
  TICK_ABOVE,
  TICK_BELOW,
  cartHeading,
  refract,
} from "../../site/lessons/light.js";
import { phase } from "../../site/lessons/moon.js";
import { GRAPH, arrivals, shortestPath } from "../../site/lessons/route.js";
import sine, { X0, dot, waveAt } from "../../site/lessons/sine.js";

const ROWS = [
  {
    path: "../../site/lessons/sine.js",
    name: "Sine wave",
    sentence: "A sine wave is a point going round a circle, watched from the side.",
    asked: "What actually is a sine wave?",
  },
  {
    path: "../../site/lessons/light.js",
    name: "Light at water",
    sentence: "Light bends at water because it slows there, like a cart with one wheel in sand.",
    asked: "Why does a straw look bent in a glass of water?",
  },
  {
    path: "../../site/lessons/route.js",
    name: "Maps route",
    sentence:
      "A maps app finds the quickest route by exploring outward from you, closest streets first.",
    asked: "How does my maps app find the fastest way home?",
  },
  {
    path: "../../site/lessons/moon.js",
    name: "Phases of the Moon",
    sentence: "The Moon's phases are just how much of its sunlit half we can see as it circles us.",
    asked: "Why does the Moon change shape?",
  },
];

const SQUARE = {
  nodes: [
    [0, 0],
    [3, 0],
    [3, 4],
    [0, 4],
  ],
  edges: [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [0, 2],
  ],
  start: 0,
  goal: 2,
};

const FIXTURE = {
  words: [["A", 0]],
  length: 11,
  asked: "Why is the sky blue?",
};

const lessons = await Promise.all(
  ROWS.map(async (row) => ({ ...row, lesson: (await import(row.path)).default })),
);

function when(lesson, word) {
  const found = lesson.words.find(([text]) => text === word);
  assert.ok(found, `no word ${word}`);
  return found[1];
}

function near(got, want, label) {
  assert.ok(Math.abs(got - want) <= 1e-9, `${label}: ${got}, not ${want}`);
}

function timed(got, want, label) {
  assert.deepEqual(got.map(([word]) => word), want.map(([word]) => word), label);
  got.forEach(([word, at], i) => near(at, want[i][1], `${label} ${word}`));
}

function steps(from, to, step) {
  const times = [];
  for (let i = 0; from + i * step < to; i++) times.push(from + i * step);
  return times;
}

function span(nodes, a, b) {
  return Math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]);
}

test("every lesson says its sentence", () => {
  for (const { path, sentence, lesson } of lessons) {
    assert.equal(lesson.words.map(([word]) => word).join(" "), sentence, path);
  }
});

test("every lesson's timing is sound", () => {
  for (const { path, lesson } of lessons) {
    const times = lesson.words.map(([, at]) => at);
    assert.equal(times[0], 0, path);
    for (let i = 1; i < times.length; i++) {
      assert.ok(times[i] > times[i - 1], `${path}: word ${i} at ${times[i]}`);
    }
    assert.ok(lesson.length >= 11 && lesson.length <= 13, `${path}: length ${lesson.length}`);
    assert.ok(lesson.length >= times.at(-1) + 3.4, `${path}: length ${lesson.length}`);
  }
});

test("exchange times a fixture lesson", () => {
  const x = exchange(FIXTURE);
  near(x.pre, 1.1, "pre");
  const asked = [
    ["Why", 0],
    ["is", 0.12],
    ["the", 0.24],
    ["sky", 0.36],
    ["blue?", 0.48],
  ];
  timed(x.asked, asked, "asked");
  near(x.total, 12.1, "total");
  assert.deepEqual(Object.keys(x).sort(), ["asked", "pre", "total"]);
});

test("every lesson asks", () => {
  for (const { path, asked, lesson } of lessons) {
    assert.equal(lesson.asked, asked, path);
    assert.deepEqual(
      Object.keys(lesson).sort(),
      ["asked", "build", "length", "name", "words"],
      path,
    );
  }
});

test("every lesson has a name", () => {
  for (const { path, name, lesson } of lessons) {
    assert.equal(typeof lesson.name, "string", path);
    assert.ok(lesson.name.length > 0, path);
    assert.equal(lesson.name, name, path);
  }
});

test("the wave leaves the dot at its height", () => {
  for (const t of steps(when(sine, "watched") + 0.05, sine.length + 1e-9, 0.1)) {
    const y = waveAt(t, X0);
    assert.notEqual(y, null, `t=${t}`);
    assert.ok(Math.abs(y - dot(t).y) <= 1e-9, `t=${t}: wave ${y}, dot ${dot(t).y}`);
  }
});

test("no wave before it is watched", () => {
  for (const t of steps(0, when(sine, "watched"), 0.1)) {
    assert.equal(waveAt(t, X0), null, `t=${t}`);
  }
});

test("the dot appears on its word", () => {
  const round = when(sine, "round");
  const rest = dot(0);
  for (const t of [...steps(0, round, 0.1), round - 0.01]) {
    assert.deepEqual(dot(t), rest, `t=${t}`);
  }
  for (const t of steps(round, sine.length, 0.01)) {
    assert.ok(dot(t).x <= rest.x + 1e-9, `t=${t}: ${dot(t).x} right of ${rest.x}`);
  }
});

test("refraction obeys snell", () => {
  assert.equal(refract(0), 0);
  for (let degrees = 10; degrees <= 80; degrees += 10) {
    const i = (degrees * Math.PI) / 180;
    const off = Math.abs(Math.sin(i) - N * Math.sin(refract(i)));
    assert.ok(off <= 1e-12, `${degrees} degrees: off by ${off}`);
  }
  const thirty = refract(Math.PI / 6);
  assert.ok(Math.abs(thirty - 0.385411) <= 1e-6, `30 degrees refracts to ${thirty}`);
});

test("ticks are closer in water", () => {
  const ratio = TICK_BELOW / TICK_ABOVE;
  assert.ok(Math.abs(ratio - 1 / N) <= 1e-12, `ratio ${ratio}`);
});

test("the cart leaves at the light's angle", () => {
  const out = refract(INCIDENCE);
  assert.equal(cartHeading(0), INCIDENCE);
  const last = cartHeading(light.length);
  assert.ok(Math.abs(last - out) <= 1e-9, `leaves at ${last}, light at ${out}`);
  let previous = cartHeading(0);
  for (const t of steps(0, light.length + 1e-9, 0.05)) {
    const heading = cartHeading(t);
    assert.ok(heading <= previous, `t=${t}: ${heading} after ${previous}`);
    previous = heading;
  }
});

test("the cart turns only while it straddles", () => {
  const one = when(light, "one");
  const out = refract(INCIDENCE);
  for (const t of [...steps(0, one, 0.05), one]) {
    assert.equal(cartHeading(t), INCIDENCE, `t=${t}`);
  }
  const done = steps(one, light.length, 0.01).find((t) => Math.abs(cartHeading(t) - out) <= 1e-12);
  assert.ok(done !== undefined, "the turn never ends");
  for (const t of steps(done, light.length + 1e-9, 0.05)) {
    assert.ok(Math.abs(cartHeading(t) - out) <= 1e-12, `t=${t}: ${cartHeading(t)}`);
  }
  const middle = cartHeading((one + done) / 2);
  assert.ok(middle < INCIDENCE && middle > out, `middle of the turn at ${middle}`);
});

test("arrivals on a square with a diagonal", () => {
  assert.deepEqual(arrivals(SQUARE), [0, 3, 5, 4]);
});

test("the diagonal is the shortest path", () => {
  assert.deepEqual(shortestPath(SQUARE), [0, 2]);
});

test("an island is unreachable", () => {
  const island = { ...SQUARE, nodes: [...SQUARE.nodes, [9, 9]] };
  assert.equal(arrivals(island)[4], Infinity);
});

test("the lesson graph meets the shortest-path conditions", () => {
  const { nodes, edges, start } = GRAPH;
  const arrival = arrivals(GRAPH);
  assert.equal(arrival[start], 0);
  arrival.forEach((at, node) => assert.ok(Number.isFinite(at), `node ${node} unreachable`));
  for (const [a, b] of edges) {
    const gap = Math.abs(arrival[a] - arrival[b]);
    assert.ok(gap <= span(nodes, a, b) + 1e-9, `street ${a}-${b}: arrivals ${gap} apart`);
  }
  arrival.forEach((at, node) => {
    if (node === start) return;
    const reached = edges.some(([a, b]) => {
      if (a !== node && b !== node) return false;
      const from = a === node ? b : a;
      return Math.abs(arrival[from] + span(nodes, a, b) - at) <= 1e-9;
    });
    assert.ok(reached, `node ${node} at ${at} has no neighbour it is reached from`);
  });
});

test("the lesson path is real and turns", () => {
  const { nodes, edges, start, goal } = GRAPH;
  const path = shortestPath(GRAPH);
  assert.equal(path[0], start);
  assert.equal(path.at(-1), goal);
  let total = 0;
  for (let i = 1; i < path.length; i++) {
    const [a, b] = [path[i - 1], path[i]];
    const street = edges.some(([u, v]) => (u === a && v === b) || (u === b && v === a));
    assert.ok(street, `no street ${a}-${b}`);
    total += span(nodes, a, b);
  }
  const arrival = arrivals(GRAPH)[goal];
  assert.ok(Math.abs(total - arrival) <= 1e-9, `path ${total}, goal reached at ${arrival}`);
  const [[x0, y0], [x1, y1]] = [nodes[path[0]], nodes[path[1]]];
  const off = path.some((node) => {
    const [x, y] = nodes[node];
    return Math.abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) > 1e-9;
  });
  assert.ok(off, `path ${path} is one straight line`);
});

test("the graph is the size the drawing needs", () => {
  const { nodes } = GRAPH;
  assert.ok(nodes.length >= 30 && nodes.length <= 45, `${nodes.length} nodes`);
  for (const [x, y] of nodes) {
    assert.ok(x >= 12 && x <= 640 - 12 && y >= 12 && y <= 300 - 12, `node at ${x},${y}`);
  }
});

test("phase at the quarters", () => {
  const quarters = [
    [0, 0],
    [Math.PI / 2, 0.5],
    [Math.PI, 1],
    [(3 * Math.PI) / 2, 0.5],
  ];
  for (const [angle, lit] of quarters) {
    const got = phase(angle).lit;
    assert.ok(Math.abs(got - lit) <= 1e-12, `angle ${angle}: lit ${got}`);
  }
  assert.equal(phase(Math.PI / 2).waxing, true);
  assert.equal(phase((3 * Math.PI) / 2).waxing, false);
});

test("lit grows while waxing and shrinks while waning", () => {
  const rising = steps(0.01, Math.PI, 0.01);
  for (let i = 1; i < rising.length; i++) {
    const [before, after] = [phase(rising[i - 1]), phase(rising[i])];
    assert.ok(after.lit > before.lit, `angle ${rising[i]}: ${after.lit} after ${before.lit}`);
    assert.equal(after.waxing, true, `angle ${rising[i]}`);
  }
  const falling = steps(Math.PI + 0.01, 2 * Math.PI, 0.01);
  for (let i = 1; i < falling.length; i++) {
    const [before, after] = [phase(falling[i - 1]), phase(falling[i])];
    assert.ok(after.lit < before.lit, `angle ${falling[i]}: ${after.lit} after ${before.lit}`);
    assert.equal(after.waxing, false, `angle ${falling[i]}`);
  }
});

test("phase wraps", () => {
  for (const angle of [0.3, 2, 4.5]) {
    assert.deepEqual(phase(angle + 2 * Math.PI), phase(angle), `angle ${angle}`);
  }
});

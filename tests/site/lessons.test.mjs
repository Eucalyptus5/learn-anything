import assert from "node:assert/strict";
import { test } from "node:test";

import sine, { X0, dot, waveAt } from "../../site/lessons/sine.js";

const ROWS = [
  {
    path: "../../site/lessons/sine.js",
    name: "Sine wave",
    sentence: "A sine wave is a point going round a circle, watched from the side.",
  },
];

const lessons = await Promise.all(
  ROWS.map(async (row) => ({ ...row, lesson: (await import(row.path)).default })),
);

function when(lesson, word) {
  const found = lesson.words.find(([text]) => text === word);
  assert.ok(found, `no word ${word}`);
  return found[1];
}

function steps(from, to, step) {
  const times = [];
  for (let i = 0; from + i * step < to; i++) times.push(from + i * step);
  return times;
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
  for (const t of steps(round, when(sine, "watched"), 0.01)) {
    assert.ok(dot(t).x <= rest.x + 1e-9, `t=${t}: ${dot(t).x} right of ${rest.x}`);
  }
});

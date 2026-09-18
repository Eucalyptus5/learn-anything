# Scene guide

A scene is one picture that teaches one idea, drawn for a learner who is listening to a tutor
and looking at the canvas. It is built once and shown for a minute or more, and the tutor
advances it one step at a time as the explanation reaches each part. The scene has to be worth
looking at for that long and has to be clear about what each step adds.

## The frame

The scene owns a frame about 800 pixels wide and 500 tall beside the tutor's words. The body is
a two-column grid: the picture in the first column, the step column the helper builds in the
second. Everything drawn sits inside one root element that fills the first column. Do not style
`body`; do not set a background on the root; the page's surface shows through.

## Type

Text inside the picture is set in the page's fonts: `var(--sans)` for labels, axes, numbers and
short lines; `var(--serif)` for a title or a sentence the learner should read as prose;
`var(--mono)` for code, a symbol name or a value the learner might type. Three sizes are enough:
13px for axis ticks and small labels, 15px for labels and callouts, 20px for the one title. Never
smaller than 12px. A label is a noun phrase, not a sentence. A number carries its unit.

## Space

Margins of 24px around the picture and 12px between things that belong together; twice that
between things that do not. Align to a few lines: axes share a left edge, labels share a
baseline, boxes in a row share a top. Leave empty space; a scene with room in it reads as
considered, a full one reads as a diagram from a manual.

## Colour

There are six variables and no other colours:

- `var(--ink)` for the main marks: the curve the scene is about, box outlines, the text that
  matters.
- `var(--ink-2)` for secondary marks and text: a second curve that is context, a caption.
- `var(--ink-3)` for the quietest: tick labels, gridlines, a note.
- `var(--accent)` for the one thing the current step is about, and for nothing else; a scene
  with two accented things has no focus.
- `var(--surface)` for a fill that has to cover something, such as a label's backing.
- `var(--hair)` for hairlines: axes, dividers, gridlines, arrow shafts that are not the point.

Never write a hex colour, a named colour or an rgb value. Never colour text for emphasis; use
weight or the accent on the mark it labels. Never use red or green to mean bad or good; the
variables have no red or green, on purpose.

## Motion

Motion shows change; it never decorates. A part appears by fading in over 0.4 to 0.6 s with
`ease: "power2.out"`. A mark moves, grows or traces over 0.6 to 1.2 s with `ease: "power2.inOut"`.
Emphasis is a single scale pulse of 1.06 over 0.5 s, once. Nothing loops, nothing bounces,
nothing spins, nothing moves that the step is not about. Text does not animate except to appear.
The accent arrives and leaves by opacity on its own element; never tween a stroke or fill toward
a variable, since var(--hair) is not a plain colour and does not interpolate.
A curve is drawn by animating `strokeDashoffset` from its length to zero, which reads as the
pen tracing it; use it once per scene at most.

The timeline is one `gsap.timeline({ paused: true })`. Each step is the span between two labels:
the label `step-n` is the state after step n has played. Add the label after the tweens that
belong to that step. Every step must add a label, so the helper can find `step-1` to `step-n`.
Step 1 is the first frame: what is on the canvas before the tutor has said anything. It is
usually the axes, the frame of the picture and its title, with the marks that later steps add
still hidden (set them `opacity: 0` in the markup and fade them in). Do not put the whole
picture in step 1 and then wiggle parts of it.

## Steps

Three to five steps. Each adds one thing the learner can name, and its say line names it in
under eighty characters, in plain words, as a label for what just appeared: `The clip band at
one plus and minus epsilon`, not `Now we add the clipping region to illustrate the bound`. The
say lines are the only prose in the scene; do not write paragraphs into the picture. The say
lines go both to `lesson.scene` and to the tool's `steps` argument, identical, in order.

## Charts

Every axis is named with its quantity and its unit, and its range is the range the brief gives
or the natural one for the quantity. Tick labels are numbers the learner would say. Use a
truncated axis only when the brief asks, and then say so on the axis. A curve is one path with
enough points to be smooth (a hundred is plenty). A point the explanation refers to gets a mark
and a label. Two curves that are compared share axes. A legend is a label beside each curve, not
a box in a corner. Plotly draws a good chart when the data is real and the axes are named; for a
chart that has to move part by part, SVG with D3 or by hand is easier to step.

## Mechanisms

A mechanism is boxes and arrows: what the parts are and what flows between them. Boxes are
rounded 6px, hairline outline, a noun inside. Arrows are hairlines with a small filled head,
labelled with what moves. Layout flows left to right or top to bottom in the order things
happen. A loop closes on itself with one return arrow. At most seven boxes. The accent goes on
the box or arrow the current step is about.

## Two scenes in full

### A chart: the clipped objective

```html
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<script src="/lesson.js"></script>
<script src="/vendor/gsap.min.js"></script>
<style>
  #scene { width: 100%; }
  svg { width: 100%; height: auto; font: 13px var(--sans); }
  .axis line, .axis path { stroke: var(--hair); }
  .axis text { fill: var(--ink-3); }
  .axis .name { fill: var(--ink-2); font-size: 15px; }
  .title { fill: var(--ink); font: 20px var(--serif); }
  .curve { fill: none; stroke: var(--ink); stroke-width: 2; }
  .band { fill: var(--accent); opacity: 0; }
  .flat { fill: none; stroke: var(--accent); stroke-width: 3; opacity: 0; }
  .note { fill: var(--ink); font-size: 15px; opacity: 0; }
</style>
</head>
<body>
<div id="scene">
<svg viewBox="0 0 560 400">
  <text class="title" x="24" y="34">Clipped objective, advantage +1, epsilon 0.2</text>
  <g class="axis" transform="translate(64 56)">
    <path d="M0 0 V300 H460" fill="none"></path>
    <text x="-8" y="304" text-anchor="end">0.5</text>
    <text x="-8" y="154" text-anchor="end">1.0</text>
    <text x="-8" y="4" text-anchor="end">1.5</text>
    <text x="0" y="318" text-anchor="middle">0.5</text>
    <text x="230" y="318" text-anchor="middle">1.0</text>
    <text x="460" y="318" text-anchor="middle">1.5</text>
    <text class="name" x="230" y="344" text-anchor="middle">probability ratio r (new over old)</text>
    <text class="name" transform="translate(-44 150) rotate(-90)" text-anchor="middle">objective L (no unit)</text>
  </g>
  <g transform="translate(64 56)">
    <rect class="band" x="138" y="0" width="184" height="300" opacity="0.10"></rect>
    <path id="curve" class="curve" d=""></path>
    <path class="flat" d="M322 90 H460"></path>
    <text class="note" x="330" y="78">flat past 1.2: no gradient</text>
  </g>
</svg>
</div>
<script>
  const x = (r) => (r - 0.5) * 460;
  const y = (l) => 300 - (l - 0.5) * 300;
  const points = [];
  for (let i = 0; i <= 100; i += 1) {
    const r = 0.5 + i / 100;
    const l = Math.min(r, 1.2);
    points.push((i === 0 ? "M" : "L") + x(r).toFixed(1) + " " + y(l).toFixed(1));
  }
  const curve = document.getElementById("curve");
  curve.setAttribute("d", points.join(" "));
  const length = curve.getTotalLength();
  curve.style.strokeDasharray = length;
  curve.style.strokeDashoffset = length;

  const tl = gsap.timeline({ paused: true });
  tl.addLabel("step-1")
    .to(curve, { strokeDashoffset: 0, duration: 1.2, ease: "power2.inOut" })
    .addLabel("step-2")
    .to(".band", { opacity: 0.10, duration: 0.5, ease: "power2.out" })
    .addLabel("step-3")
    .to(".flat", { opacity: 1, duration: 0.5, ease: "power2.out" })
    .to(".note", { opacity: 1, duration: 0.4, ease: "power2.out" }, "<0.2")
    .addLabel("step-4");

  lesson.scene({
    root: "#scene",
    timeline: tl,
    steps: [
      "Axes: the ratio against the objective",
      "The objective rises with the ratio",
      "The band from 0.8 to 1.2 is one plus and minus epsilon",
      "Past 1.2 the objective is flat, so there is no gradient",
    ],
  });
</script>
</body>
</html>
```

The band is in the markup at opacity 0 and fades to a tenth; the curve is traced once; the
accent is on the band and the flat segment, which are what the last two steps are about; the
axes carry their quantity, and the unit line says the objective has none.

### A mechanism: one PPO iteration

```html
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<script src="/lesson.js"></script>
<script src="/vendor/gsap.min.js"></script>
<style>
  #scene { width: 100%; }
  svg { width: 100%; height: auto; font: 15px var(--sans); }
  .box { fill: none; stroke: var(--hair); stroke-width: 1.5; rx: 6; }
  .focus { fill: none; stroke: var(--accent); stroke-width: 2.5; rx: 6; opacity: 0; }
  .label { fill: var(--ink); text-anchor: middle; }
  .flow { fill: none; stroke: var(--hair); stroke-width: 1.5; marker-end: url(#head); }
  .flow-label { fill: var(--ink-3); font-size: 13px; text-anchor: middle; }
  .title { fill: var(--ink); font: 20px var(--serif); }
  .later { opacity: 0; }
</style>
</head>
<body>
<div id="scene">
<svg viewBox="0 0 560 400">
  <defs>
    <marker id="head" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="8" markerHeight="8" orient="auto">
      <path d="M0 0 L8 4 L0 8 z" fill="var(--ink-3)"></path>
    </marker>
  </defs>
  <text class="title" x="24" y="34">One PPO iteration</text>
  <g id="rollout">
    <rect class="box" x="40" y="120" width="140" height="56"></rect>
    <text class="label" x="110" y="154">collect rollouts</text>
  </g>
  <g id="advantage" class="later">
    <path class="flow" d="M180 148 H236"></path>
    <text class="flow-label" x="208" y="138">states, rewards</text>
    <rect class="box" x="236" y="120" width="140" height="56"></rect>
    <text class="label" x="306" y="154">estimate advantage</text>
  </g>
  <g id="update" class="later">
    <path class="flow" d="M376 148 H432"></path>
    <text class="flow-label" x="404" y="138">A(s, a)</text>
    <rect class="box" x="432" y="120" width="104" height="56"></rect>
    <rect class="focus" x="432" y="120" width="104" height="56"></rect>
    <text class="label" x="484" y="154">clipped update</text>
  </g>
  <g id="epochs" class="later">
    <path class="flow" d="M484 176 V236 H110 V176"></path>
    <text class="flow-label" x="297" y="258">the updated policy collects the next batch</text>
  </g>
</svg>
</div>
<script>
  const tl = gsap.timeline({ paused: true });
  tl.addLabel("step-1")
    .to("#advantage", { opacity: 1, duration: 0.5, ease: "power2.out" })
    .addLabel("step-2")
    .to("#update", { opacity: 1, duration: 0.5, ease: "power2.out" })
    .to("#update .focus", { opacity: 1, duration: 0.4, ease: "power2.out" }, "<")
    .addLabel("step-3")
    .to("#update .focus", { opacity: 0, duration: 0.4, ease: "power2.out" })
    .to("#epochs", { opacity: 1, duration: 0.6, ease: "power2.out" }, "<")
    .addLabel("step-4");

  lesson.scene({
    root: "#scene",
    timeline: tl,
    steps: [
      "The current policy collects a batch of rollouts",
      "Each action gets an advantage estimate from the batch",
      "Several epochs of the clipped update on that batch",
      "The updated policy becomes the old one and the loop repeats",
    ],
  });
</script>
</body>
</html>
```

The first frame is one box; each step adds the next part and its arrow; the accent sits on the
update box only while step 3 is about it and leaves when the loop closes.

const SENTENCE =
  "A maps app finds the quickest route by exploring outward from you, closest streets first.";
const NS = "http://www.w3.org/2000/svg";

const words = [];
let next = 0;
for (const word of SENTENCE.split(" ")) {
  words.push([word, Math.round(next * 100) / 100]);
  next += word.endsWith(",") ? 0.32 : 0.2;
}

function when(word) {
  return words.find(([text]) => text === word)[1];
}

const GRID_START = when("A");
const EXPLORING = when("exploring");

export const GRAPH = {
  nodes: [
    [50, 40], [131, 43], [209, 42], [285, 38], [359, 36], [432, 36], [508, 39], [587, 40],
    [54, 92], [131, 98], [205, 102], [278, 101], [353, 97], [431, 93], [511, 91], [593, 92],
    [51, 145], [125, 149], [199, 155], [275, 159], [354, 159], [435, 155], [517, 149], [598, 145],
    [45, 206], [120, 205], [198, 206], [278, 211], [360, 215], [441, 216], [520, 212], [596, 206],
    [42, 268], [121, 267], [203, 265], [285, 264], [365, 266], [442, 269], [517, 271], [590, 268],
  ],
  edges: [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
    [16, 17], [17, 18], [18, 19], [19, 20], [21, 22], [22, 23],
    [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31],
    [32, 33], [33, 34], [34, 35], [37, 38],
    [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15],
    [8, 16], [9, 17], [10, 18], [11, 19], [12, 20], [13, 21], [14, 22], [15, 23],
    [16, 24], [17, 25], [18, 26], [19, 27], [20, 28], [21, 29], [22, 30], [23, 31],
    [24, 32], [25, 33], [26, 34], [27, 35], [28, 36], [29, 37], [30, 38], [31, 39],
    [27, 20], [9, 2],
  ],
  start: 25,
  goal: 14,
};

function span(nodes, a, b) {
  return Math.hypot(nodes[b][0] - nodes[a][0], nodes[b][1] - nodes[a][1]);
}

function search({ nodes, edges, start }) {
  const arrival = nodes.map(() => Infinity);
  const previous = nodes.map(() => -1);
  const settled = nodes.map(() => false);
  arrival[start] = 0;
  for (;;) {
    let near = -1;
    arrival.forEach((at, node) => {
      if (!settled[node] && at < Infinity && (near < 0 || at < arrival[near])) near = node;
    });
    if (near < 0) return { arrival, previous };
    settled[near] = true;
    for (const [a, b] of edges) {
      if (a !== near && b !== near) continue;
      const other = a === near ? b : a;
      const via = arrival[near] + span(nodes, a, b);
      if (via < arrival[other]) {
        arrival[other] = via;
        previous[other] = near;
      }
    }
  }
}

export function arrivals(graph) {
  return search(graph).arrival;
}

export function shortestPath(graph) {
  const { previous } = search(graph);
  const path = [];
  for (let node = graph.goal; node >= 0; node = previous[node]) path.unshift(node);
  return path;
}

const { nodes: NODES, edges: EDGES } = GRAPH;
const ARRIVAL = arrivals(GRAPH);
const PATH = shortestPath(GRAPH);
const GOAL_DISTANCE = ARRIVAL[GRAPH.goal];

const FLOOD = 3.65;
const REACH = EXPLORING + FLOOD;
const YIELD = 0.5;
const DRAW = 0.6;
const APPEAR = 0.25;
const ROUTE_WIDTH = 4.5;

const XS = NODES.map(([x]) => x);
const LEFT = Math.min(...XS);
const RIGHT = Math.max(...XS);

function sketched(x) {
  return GRID_START + ((x - LEFT) / (RIGHT - LEFT)) * (EXPLORING - GRID_START - DRAW);
}

const STREETS = EDGES.map(([a, b]) => {
  const [p, q] = [NODES[a], NODES[b]];
  const [from, to] = p[0] + p[1] <= q[0] + q[1] ? [p, q] : [q, p];
  return { from, to, at: sketched(from[0]) };
});

function ramp(value, at, over) {
  return Math.min(1, Math.max(0, (value - at) / over));
}

function along(p, q, share) {
  return [p[0] + (q[0] - p[0]) * share, p[1] + (q[1] - p[1]) * share];
}

function xy([x, y]) {
  return `${x.toFixed(2)},${y.toFixed(2)}`;
}

function stretch(line, from, to, share) {
  const [x, y] = along(from, to, share);
  line.setAttribute("x1", from[0]);
  line.setAttribute("y1", from[1]);
  line.setAttribute("x2", x.toFixed(2));
  line.setAttribute("y2", y.toFixed(2));
  line.setAttribute("visibility", share > 0 ? "visible" : "hidden");
}

export default {
  name: "Maps route",
  words,
  length: 11,
  asked: "How does my maps app find the fastest way home?",
  askBack: "So which streets would it check last?",
  askAt: 6.15,
  build(svg) {
    const doc = svg.ownerDocument;
    const group = doc.createElementNS(NS, "g");
    const make = (tag, attributes, parent = group) => {
      const element = doc.createElementNS(NS, tag);
      for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
      parent.append(element);
      return element;
    };

    const grid = make("path", { class: "construct" });
    const flood = make("g", {});
    const fills = EDGES.map(() => [
      make("line", { class: "result" }, flood),
      make("line", { class: "result" }, flood),
    ]);
    // The class's stroke-width outranks an attribute; only an inline style can widen it.
    const route = make("path", { class: "result", style: `stroke-width: ${ROUTE_WIDTH}px` });
    const [goalX, goalY] = NODES[GRAPH.goal];
    const ring = make("circle", { class: "result", cx: goalX, cy: goalY, r: 11 });
    const [startX, startY] = NODES[GRAPH.start];
    const you = make("circle", { class: "point", cx: startX, cy: startY, r: 7 });
    svg.append(group);

    return (t) => {
      const sketch = [];
      for (const { from, to, at } of STREETS) {
        const share = ramp(t, at, DRAW);
        if (share > 0) sketch.push(`M${xy(from)} L${xy(along(from, to, share))}`);
      }
      grid.setAttribute("d", sketch.join(" "));
      ring.setAttribute("opacity", ramp(t, sketched(goalX), APPEAR));
      you.setAttribute("opacity", ramp(t, sketched(startX), APPEAR));

      const front = GOAL_DISTANCE * ramp(t, EXPLORING, FLOOD);
      EDGES.forEach(([a, b], i) => {
        const length = span(NODES, a, b);
        stretch(fills[i][0], NODES[a], NODES[b], ramp(front, ARRIVAL[a], length));
        stretch(fills[i][1], NODES[b], NODES[a], ramp(front, ARRIVAL[b], length));
      });
      flood.setAttribute("opacity", 1 - ramp(t, REACH, YIELD));

      const traced = GOAL_DISTANCE * ramp(t, REACH, YIELD);
      const legs = [];
      for (let i = 1; i < PATH.length; i++) {
        const [a, b] = [PATH[i - 1], PATH[i]];
        const share = ramp(traced, ARRIVAL[a], ARRIVAL[b] - ARRIVAL[a]);
        if (share > 0) legs.push(xy(along(NODES[a], NODES[b], share)));
      }
      route.setAttribute("d", legs.length > 0 ? `M${xy(NODES[PATH[0]])} L${legs.join(" L")}` : "");
    };
  },
};

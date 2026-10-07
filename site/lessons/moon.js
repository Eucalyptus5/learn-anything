const SENTENCE =
  "The Moon's phases are just how much of its sunlit half we can see as it circles us.";
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

const MOON = when("Moon's");
const SUNLIT = when("sunlit");
const SEE = when("see");
const CIRCLES = when("circles");

const TAU = 2 * Math.PI;

export function phase(angle) {
  const turn = ((angle % TAU) + TAU) % TAU;
  return { lit: (1 - Math.cos(turn)) / 2, waxing: turn > 0 && turn < Math.PI };
}

const EARTH = { x: 200, y: 150 };
const EARTH_R = 16;
const ORBIT_R = 90;
const MOON_R = 11;
const CLEAR = 4;
const GAP = 2 * Math.asin((MOON_R + CLEAR) / (2 * ORBIT_R));
const NEAR = EARTH_R + CLEAR;
const FAR = ORBIT_R - MOON_R - CLEAR;

const RAYS = [70, 110, 150, 190, 230];
const RAY_FROM = 14;
const RAY_TO = 80;
const HEAD = 7;
const ARROW = `m${-HEAD},${-HEAD} l${HEAD},${HEAD} l${-HEAD},${HEAD}`;

const INSET = { x: 472, y: 118, r: 80 };
const ROW_Y = 252;
const COPY_R = 16;
const PITCH = 39;

const SHINE = 0.8;
const SKETCH = 1.2;
const FILL = 0.4;
const SIGHT = 0.3;
const SHOW = 0.4;
const TURN = 4;
const APPEAR = 0.25;

const COPIES = [1, 2, 3, 4, 5, 6, 7, 8].map((k) => ({
  angle: (k * TAU) / 8,
  at: CIRCLES + (k * TURN) / 8,
  x: INSET.x + (k - 4.5) * PITCH,
}));

function ramp(t, at, over) {
  return Math.min(1, Math.max(0, (t - at) / over));
}

function around(angle, radius) {
  return { x: EARTH.x - radius * Math.cos(angle), y: EARTH.y + radius * Math.sin(angle) };
}

function xy({ x, y }) {
  return `${x.toFixed(2)},${y.toFixed(2)}`;
}

function seen(x, y, r, angle) {
  const { lit, waxing } = phase(angle);
  if (lit === 0) return "";
  const bulge = (waxing ? 1 : -1) * r * (1 - 2 * lit);
  const top = xy({ x, y: y - r });
  const bottom = xy({ x, y: y + r });
  const limb = `A${r},${r} 0 0 ${waxing ? 1 : 0} ${bottom}`;
  const terminator = `A${Math.abs(bulge).toFixed(2)},${r} 0 0 ${bulge > 0 ? 0 : 1} ${top}`;
  return `M${top} ${limb} ${terminator} Z`;
}

export default {
  name: "Phases of the Moon",
  words,
  length: 11,
  build(svg) {
    const doc = svg.ownerDocument;
    const group = doc.createElementNS(NS, "g");
    const make = (tag, attributes, parent = group) => {
      const element = doc.createElementNS(NS, tag);
      for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
      parent.append(element);
      return element;
    };

    const rays = make("path", { class: "construct" });
    const earth = make("circle", {
      class: "construct faint",
      cx: EARTH.x,
      cy: EARTH.y,
      r: EARTH_R,
    });
    const orbit = make("path", { class: "construct" });
    const sightline = make("line", { class: "construct dash" });
    const moon = make("g", {});
    make("circle", { class: "construct faint", r: MOON_R }, moon);
    const half = make(
      "path",
      { class: "lit", d: `M0,${-MOON_R} A${MOON_R},${MOON_R} 0 0 0 0,${MOON_R} Z` },
      moon,
    );
    const inset = make("g", {});
    make("circle", { class: "construct faint", cx: INSET.x, cy: INSET.y, r: INSET.r }, inset);
    const face = make("path", { class: "lit" }, inset);
    const copies = COPIES.map(({ angle, x }) => {
      const copy = make("g", {});
      make("circle", { class: "construct faint", cx: x, cy: ROW_Y, r: COPY_R }, copy);
      make("path", { class: "lit", d: seen(x, ROW_Y, COPY_R, angle) }, copy);
      return copy;
    });
    svg.append(group);

    return (t) => {
      const tip = RAY_FROM + (RAY_TO - RAY_FROM) * ramp(t, 0, SHINE);
      const beams = RAYS.map((y) => `M${RAY_FROM},${y} L${xy({ x: tip, y })} ${ARROW}`);
      rays.setAttribute("d", tip > RAY_FROM ? beams.join(" ") : "");
      earth.setAttribute("opacity", ramp(t, 0, APPEAR));

      const angle = TAU * ramp(t, CIRCLES, TURN);
      const span = (TAU - 2 * GAP) * ramp(t, MOON, SKETCH);
      const start = angle + GAP;
      // Under a quarter turn each: a half-turn arc between rounded endpoints shifts its centre.
      const arcs = [1, 2, 3, 4].map(
        (i) => `A${ORBIT_R},${ORBIT_R} 0 0 0 ${xy(around(start + (span * i) / 4, ORBIT_R))}`,
      );
      orbit.setAttribute("d", span > 0 ? `M${xy(around(start, ORBIT_R))} ${arcs.join(" ")}` : "");

      moon.setAttribute("transform", `translate(${xy(around(angle, ORBIT_R))})`);
      moon.setAttribute("opacity", ramp(t, MOON, APPEAR));
      half.setAttribute("opacity", ramp(t, SUNLIT, FILL));

      const reach = ramp(t, SEE, SIGHT);
      const from = around(angle, NEAR);
      const to = around(angle, NEAR + (FAR - NEAR) * reach);
      sightline.setAttribute("x1", from.x.toFixed(2));
      sightline.setAttribute("y1", from.y.toFixed(2));
      sightline.setAttribute("x2", to.x.toFixed(2));
      sightline.setAttribute("y2", to.y.toFixed(2));
      sightline.setAttribute("visibility", reach > 0 ? "visible" : "hidden");

      inset.setAttribute("opacity", ramp(t, SEE, SHOW));
      face.setAttribute("d", seen(INSET.x, INSET.y, INSET.r, angle));
      COPIES.forEach(({ at }, i) => copies[i].setAttribute("opacity", ramp(t, at, APPEAR)));
    };
  },
};

const SENTENCE = "Light bends at water because it slows there, like a cart with one wheel in sand.";
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

const LIGHT = when("Light");
const BENDS = when("bends");
const SLOWS = when("slows");
const CART = when("cart");
const ONE = when("one");

export const N = 1.33;
export const INCIDENCE = (50 * Math.PI) / 180;

export function refract(incidence) {
  return Math.asin(Math.sin(incidence) / N);
}

const REFRACTED = refract(INCIDENCE);
const IN = { x: Math.sin(INCIDENCE), y: Math.cos(INCIDENCE) };
const OUT = { x: Math.sin(REFRACTED), y: Math.cos(REFRACTED) };

const WATER = 125;
const RAY_SPEED = 120;
const TICK_EVERY = 0.2;
export const TICK_ABOVE = RAY_SPEED * TICK_EVERY;
export const TICK_BELOW = (RAY_SPEED / N) * TICK_EVERY;
const STEPS_ABOVE = 7;
const STEPS_BELOW = 10;
const ABOVE = STEPS_ABOVE * TICK_EVERY;
const BELOW = (STEPS_BELOW + 0.5) * TICK_EVERY;
const CROSS = { x: 225, y: WATER };
const START = {
  x: CROSS.x - STEPS_ABOVE * TICK_ABOVE * IN.x,
  y: CROSS.y - STEPS_ABOVE * TICK_ABOVE * IN.y,
};
const TICK = 5;
const NORMAL = 60;

const AXLE = 36;
const CART_SPEED = 18;
const SWIVEL = (CART_SPEED * (1 - 1 / N)) / AXLE;
const SWIVEL_RADIUS = (CART_SPEED * (1 + 1 / N)) / 2 / SWIVEL;
// The trailing wheel reaches the line just as the heading reaches Snell's angle.
const STRAIGHTEN = ONE + (INCIDENCE - REFRACTED) / SWIVEL;
const LANDING = { x: 420, y: WATER - (AXLE / 2) * IN.x };
const WHEEL = { long: 13, wide: 6 };

const APPEAR = 0.25;

export function cartHeading(t) {
  return Math.max(REFRACTED, INCIDENCE - SWIVEL * Math.max(0, t - ONE));
}

function rayAt(time) {
  const s = Math.min(Math.max(0, time), ABOVE + BELOW);
  if (s <= ABOVE) {
    return { x: START.x + RAY_SPEED * s * IN.x, y: START.y + RAY_SPEED * s * IN.y };
  }
  const below = (RAY_SPEED / N) * (s - ABOVE);
  return { x: CROSS.x + below * OUT.x, y: CROSS.y + below * OUT.y };
}

function cartAt(t) {
  if (t <= ONE) {
    const run = CART_SPEED * (t - ONE);
    return { x: LANDING.x + run * IN.x, y: LANDING.y + run * IN.y };
  }
  const heading = cartHeading(t);
  const run = (CART_SPEED / N) * Math.max(0, t - STRAIGHTEN);
  return {
    x: LANDING.x + SWIVEL_RADIUS * (Math.cos(heading) - IN.y) + run * OUT.x,
    y: LANDING.y + SWIVEL_RADIUS * (IN.x - Math.sin(heading)) + run * OUT.y,
  };
}

const TRAIL = {
  x: LANDING.x - ((LANDING.y - START.y) / IN.y) * IN.x,
  y: START.y,
};

const TICKS = [];
for (let k = 1; k <= STEPS_ABOVE + STEPS_BELOW; k++) {
  if (k !== STEPS_ABOVE) TICKS.push(k * TICK_EVERY);
}

function xy({ x, y }) {
  return `${x.toFixed(2)},${y.toFixed(2)}`;
}

function appear(t, at) {
  return Math.min(1, Math.max(0, (t - at) / APPEAR));
}

export default {
  name: "Light at water",
  words,
  length: 12,
  build(svg) {
    const doc = svg.ownerDocument;
    const group = doc.createElementNS(NS, "g");
    const make = (tag, attributes, parent = group) => {
      const element = doc.createElementNS(NS, tag);
      for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
      parent.append(element);
      return element;
    };

    const defs = make("defs", {});
    const fade = make("radialGradient", { id: "light-fade", cy: WATER / 300 }, defs);
    make("stop", { offset: 0.6, "stop-opacity": 1 }, fade);
    make("stop", { offset: 1, "stop-opacity": 0 }, fade);
    // An alpha mask reads only stop-opacity, so the fade carries no colour.
    const mask = make("mask", { id: "light-shore", "mask-type": "alpha" }, defs);
    make("rect", { width: 640, height: 300, fill: "url(#light-fade)" }, mask);
    const shore = make("g", { mask: "url(#light-shore)" });
    make("rect", { class: "faint", x: 0, y: WATER, width: 640, height: 300 - WATER }, shore);
    make("line", { class: "construct", x1: 0, y1: WATER, x2: 640, y2: WATER }, shore);
    const normal = make("line", {
      class: "construct dash",
      x1: CROSS.x,
      y1: WATER - NORMAL,
      x2: CROSS.x,
      y2: WATER + NORMAL,
    });
    const trail = make("path", { class: "construct" });
    const ray = make("path", { class: "result" });
    const ticks = TICKS.map((time) => {
      const at = rayAt(time);
      const along = time < ABOVE ? IN : OUT;
      return make("line", {
        class: "result",
        x1: (at.x - TICK * along.y).toFixed(2),
        y1: (at.y + TICK * along.x).toFixed(2),
        x2: (at.x + TICK * along.y).toFixed(2),
        y2: (at.y - TICK * along.x).toFixed(2),
      });
    });
    const cart = make("g", {});
    make("line", { class: "result", x1: -AXLE / 2, y1: 0, x2: AXLE / 2, y2: 0 }, cart);
    for (const side of [-1, 1]) {
      make(
        "rect",
        {
          class: "point",
          x: (side * AXLE - WHEEL.wide) / 2,
          y: -WHEEL.long / 2,
          width: WHEEL.wide,
          height: WHEEL.long,
          rx: WHEEL.wide / 3,
        },
        cart,
      );
    }
    svg.append(group);

    return (t) => {
      shore.setAttribute("opacity", appear(t, 0));
      normal.setAttribute("opacity", appear(t, BENDS));

      const head = t - LIGHT;
      const bend = head > ABOVE ? ` L${xy(CROSS)}` : "";
      ray.setAttribute("d", head > 0 ? `M${xy(START)}${bend} L${xy(rayAt(head))}` : "");
      TICKS.forEach((time, i) => {
        ticks[i].setAttribute("opacity", appear(t, Math.max(SLOWS, LIGHT + time)));
      });

      const center = cartAt(t);
      const degrees = (-cartHeading(t) * 180) / Math.PI;
      cart.setAttribute("transform", `translate(${xy(center)}) rotate(${degrees.toFixed(3)})`);
      cart.setAttribute("opacity", appear(t, CART));

      const path = [`M${xy(TRAIL)} L${xy(cartAt(Math.min(t, ONE)))}`];
      if (t > ONE) {
        const radius = SWIVEL_RADIUS.toFixed(2);
        path.push(`A${radius},${radius} 0 0 1 ${xy(cartAt(Math.min(t, STRAIGHTEN)))}`);
      }
      if (t > STRAIGHTEN) path.push(`L${xy(center)}`);
      trail.setAttribute("d", path.join(" "));
      trail.setAttribute("opacity", appear(t, CART));
    };
  },
};

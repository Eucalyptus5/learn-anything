const SENTENCE = "A sine wave is a point going round a circle, watched from the side.";
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

const POINT = when("point");
const ROUND = when("round");
const WATCHED = when("watched");

const CX = 100;
const CY = 150;
const R = 90;
export const X0 = 260;
const X1 = 630;
const SPEED = 90;
const TURN = 1.45;
const OMEGA = (2 * Math.PI) / TURN;
const APPEAR = 0.25;

export function dot(t) {
  const angle = OMEGA * Math.max(0, t - ROUND);
  return { x: CX + R * Math.cos(angle), y: CY - R * Math.sin(angle) };
}

function reach(t) {
  return Math.min(X1, X0 + SPEED * (t - WATCHED));
}

export function waveAt(t, x) {
  if (t < WATCHED || x < X0 || x > reach(t)) return null;
  return dot(t - (x - X0) / SPEED).y;
}

function onCircle(angle) {
  return `${(CX + R * Math.cos(angle)).toFixed(2)},${(CY - R * Math.sin(angle)).toFixed(2)}`;
}

function appear(t, at) {
  return Math.min(1, Math.max(0, (t - at) / APPEAR));
}

export default {
  name: "Sine wave",
  words,
  length: 11,
  asked: "What actually is a sine wave?",
  build(svg) {
    const doc = svg.ownerDocument;
    const group = doc.createElementNS(NS, "g");
    const make = (tag, className) => {
      const element = doc.createElementNS(NS, tag);
      element.setAttribute("class", className);
      group.append(element);
      return element;
    };
    const circle = make("path", "construct");
    const level = make("line", "construct dash");
    const wave = make("path", "result");
    const point = make("circle", "point");
    point.setAttribute("r", "5");
    svg.append(group);

    return (t) => {
      const swept = Math.min(2 * Math.PI, OMEGA * Math.max(0, t - ROUND));
      const half = `M${onCircle(0)} A${R},${R} 0 0 0 ${onCircle(Math.min(swept, Math.PI))}`;
      const rest = swept > Math.PI ? ` A${R},${R} 0 0 0 ${onCircle(swept)}` : "";
      circle.setAttribute("d", swept > 0 ? half + rest : "");

      const { x, y } = dot(t);
      point.setAttribute("cx", x.toFixed(2));
      point.setAttribute("cy", y.toFixed(2));
      point.setAttribute("opacity", appear(t, POINT));

      level.setAttribute("x1", x.toFixed(2));
      level.setAttribute("y1", y.toFixed(2));
      level.setAttribute("x2", X0);
      level.setAttribute("y2", y.toFixed(2));
      level.setAttribute("opacity", appear(t, WATCHED));

      if (t < WATCHED) {
        wave.setAttribute("d", "");
        return;
      }
      const end = reach(t);
      const points = [];
      for (let sample = X0; sample < end; sample += 2) {
        points.push(`${sample},${waveAt(t, sample).toFixed(2)}`);
      }
      points.push(`${end.toFixed(2)},${waveAt(t, end).toFixed(2)}`);
      wave.setAttribute("d", `M${points.join(" L")}`);
    };
  },
};

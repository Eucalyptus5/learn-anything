(() => {
  const STYLE = `
:root {
  --ink: #17181b; --ink-2: #5f6368; --ink-3: #9a9ea6; --accent: #0e7c73; --surface: #ffffff;
  --hair: color-mix(in srgb, var(--ink) 9%, transparent);
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --serif: Charter, "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  --mono: ui-monospace, Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #ebeae6; --ink-2: #a6aab2; --ink-3: #6e737c; --accent: #63d3c4; --surface: #1a1b1f;
    --hair: color-mix(in srgb, var(--ink) 8%, transparent);
    color-scheme: dark;
  }
}
body {
  margin: 0; padding: 16px; display: grid; grid-template-columns: 1fr 220px; gap: 20px;
  background: var(--surface); color: var(--ink); font: 14px/1.5 var(--sans);
}
.lesson-steps {
  grid-column: 2; grid-row: 1; align-self: start; margin: 0; padding-left: 16px;
  border-left: 1px solid var(--hair);
}
.lesson-steps p { margin: 0 0 10px; opacity: 0; color: var(--ink-3); }
.lesson-steps p.shown { opacity: 1; }
.lesson-steps p.now { color: var(--ink); }
`;
  const STROKED = new Set(["path", "line", "polyline", "polygon", "circle", "ellipse", "rect"]);
  const FIRST_MS = 400;
  const BEAT_MS = 1600;
  const BEAT_PER_CHAR_MS = 40;
  const BEAT_CAP_MS = 5000;
  const FADE_MS = 350;
  const DRAW_MS = 700;
  const EASE = "ease-out";
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hidden = new Map();
  let played = false;

  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.append(style);

  function beat(say) {
    return Math.min(BEAT_CAP_MS, BEAT_MS + BEAT_PER_CHAR_MS * say.length);
  }

  function matched(selector) {
    if (typeof selector !== "string") return [];
    try {
      return [...document.querySelectorAll(selector)];
    } catch (error) {
      console.error("lesson: bad selector " + selector, error);
      return [];
    }
  }

  function targets(step) {
    return new Set([...step.targets, ...matched(step.show)]);
  }

  function hide(el) {
    if (hidden.has(el)) return;
    hidden.set(el, {
      opacity: el.style.opacity,
      dasharray: el.style.strokeDasharray,
      dashoffset: el.style.strokeDashoffset,
    });
    el.style.opacity = "0";
  }

  function restore(el, own) {
    el.style.opacity = own.opacity;
    el.style.strokeDasharray = own.dasharray;
    el.style.strokeDashoffset = own.dashoffset;
  }

  function stroked(el) {
    return (
      STROKED.has(el.tagName.toLowerCase()) &&
      getComputedStyle(el).fill === "none" &&
      typeof el.getTotalLength === "function"
    );
  }

  function settle(anim, el, own) {
    anim.onfinish = () => {
      anim.cancel();
      restore(el, own);
    };
  }

  function reveal(el) {
    const own = hidden.get(el);
    if (own === undefined) return;
    hidden.delete(el);
    if (still) {
      restore(el, own);
      return;
    }
    el.style.opacity = own.opacity;
    if (stroked(el)) {
      const length = el.getTotalLength();
      el.style.strokeDasharray = String(length);
      el.style.strokeDashoffset = String(length);
      const anim = el.animate([{ strokeDashoffset: length }, { strokeDashoffset: 0 }], {
        duration: DRAW_MS,
        easing: EASE,
        fill: "forwards",
      });
      settle(anim, el, own);
      return;
    }
    const frames =
      el instanceof SVGElement
        ? [{ opacity: 0 }, { opacity: 1 }]
        : [{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }];
    settle(el.animate(frames, { duration: FADE_MS, easing: EASE, fill: "forwards" }), el, own);
  }

  function run(step) {
    if (typeof step.run !== "function") return;
    try {
      step.run();
    } catch (error) {
      console.error("lesson: step run failed", error);
    }
  }

  function line(p, column) {
    column.querySelector("p.now")?.classList.remove("now");
    p.classList.add("shown", "now");
    if (still) return;
    const anim = p.animate(
      [{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }],
      { duration: FADE_MS, easing: EASE, fill: "forwards" },
    );
    anim.onfinish = () => anim.cancel();
  }

  function play(step, p, column) {
    line(p, column);
    run(step);
    for (const el of targets(step)) reveal(el);
  }

  function steps(list) {
    if (played || !Array.isArray(list)) return;
    played = true;
    const plan = list
      .filter((step) => step !== null && typeof step === "object" && typeof step.say === "string")
      .map((step) => ({ say: step.say, show: step.show, run: step.run, targets: matched(step.show) }));
    const column = document.createElement("aside");
    column.className = "lesson-steps";
    const lines = plan.map((step) => {
      const p = document.createElement("p");
      p.textContent = step.say;
      column.append(p);
      return p;
    });
    document.body.append(column);
    for (const step of plan) for (const el of step.targets) hide(el);
    if (still) {
      plan.forEach((step, i) => {
        lines[i].classList.add("shown", "now");
        run(step);
        for (const el of targets(step)) reveal(el);
      });
      return;
    }
    let at = FIRST_MS;
    plan.forEach((step, i) => {
      setTimeout(() => play(step, lines[i], column), at);
      at += beat(step.say);
    });
  }

  window.lesson = Object.freeze({ steps });
})();

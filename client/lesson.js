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
.lesson-steps p { margin: 0 0 10px; opacity: 0; color: var(--ink-3); transition: opacity 350ms ease-out; }
.lesson-steps p.shown { opacity: 1; }
.lesson-steps p.now { color: var(--ink); }
@media (prefers-reduced-motion: reduce) { .lesson-steps p { transition: none; } }
`;
  const REPORT_MS = 5000;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let current = null;
  let lines = [];
  let error = "";
  let reported = false;

  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.append(style);

  function note(message) {
    if (error === "") error = String(message).slice(0, 500);
  }

  window.addEventListener("error", (event) => note(event.message));
  window.addEventListener("unhandledrejection", (event) => note(event.reason));

  function report(steps, root) {
    if (reported) return;
    reported = true;
    const box = root === null ? { width: 0, height: 0 } : root.getBoundingClientRect();
    window.parent.postMessage(
      { type: "scene.ready", steps, width: box.width, height: box.height, error },
      "*",
    );
  }

  function label(n) {
    return "step-" + n;
  }

  function mark(n) {
    lines.forEach((p, i) => {
      p.classList.toggle("shown", i < n);
      p.classList.toggle("now", i === n - 1);
    });
  }

  function go(n) {
    mark(n);
    if (still) current.timeline.seek(label(n));
    else current.timeline.tweenTo(label(n));
  }

  function register(spec) {
    if (current !== null) throw new Error("lesson.scene was called twice");
    if (typeof spec !== "object" || spec === null) throw new Error("lesson.scene takes an object");
    const root = typeof spec.root === "string" ? document.querySelector(spec.root) : spec.root;
    if (!(root instanceof Element)) throw new Error("root is not an element in the document");
    const timeline = spec.timeline;
    if (
      typeof timeline !== "object" ||
      timeline === null ||
      typeof timeline.tweenTo !== "function" ||
      typeof timeline.labels !== "object"
    ) {
      throw new Error("timeline is not a gsap timeline");
    }
    const steps = Array.isArray(spec.steps) ? spec.steps : [];
    if (steps.length === 0 || !steps.every((say) => typeof say === "string" && say !== "")) {
      throw new Error("steps is not a list of non-empty strings");
    }
    const missing = steps.map((_, i) => label(i + 1)).filter((name) => !(name in timeline.labels));
    if (missing.length > 0) throw new Error("timeline lacks labels " + missing.join(" "));
    const column = document.createElement("aside");
    column.className = "lesson-steps";
    lines = steps.map((say) => {
      const p = document.createElement("p");
      p.textContent = say;
      column.append(p);
      return p;
    });
    document.body.append(column);
    current = { root, timeline, steps };
    timeline.pause().seek(label(1));
    mark(1);
  }

  function scene(spec) {
    try {
      register(spec);
    } catch (failure) {
      note(failure.message);
    }
    requestAnimationFrame(function settle() {
      if (reported) return;
      if (current !== null && current.root.getBoundingClientRect().width === 0) {
        requestAnimationFrame(settle);
        return;
      }
      report(current === null ? 0 : current.steps.length, current === null ? null : current.root);
    });
  }

  window.addEventListener("message", (event) => {
    const m = event.data;
    if (current === null || typeof m !== "object" || m === null || !Number.isInteger(m.step)) return;
    if (m.step < 1 || m.step > current.steps.length) return;
    go(m.step);
  });

  function fallback() {
    if (reported) return;
    if (document.visibilityState === "hidden") {
      document.addEventListener("visibilitychange", () => setTimeout(fallback, REPORT_MS), { once: true });
      return;
    }
    if (current === null) note("lesson.scene was not called within " + REPORT_MS + " ms");
    report(current === null ? 0 : current.steps.length, current === null ? null : current.root);
  }

  setTimeout(fallback, REPORT_MS);

  window.lesson = Object.freeze({ scene });
})();

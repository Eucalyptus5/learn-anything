export const ASK_STEP = 0.12;
export const GAP = 0.5;
export const FADE = 0.8;

export function exchange(lesson) {
  const asked = lesson.asked.split(" ").map((word, i) => [word, i * ASK_STEP]);
  const pre = asked.length * ASK_STEP + GAP;
  return { pre, asked, total: pre + lesson.length };
}

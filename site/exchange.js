export const ASK_STEP = 0.12;
export const GAP = 0.5;
export const SWAP = 0.4;
export const FADE = 0.8;
const LISTEN = 0.45;

export function exchange(lesson) {
  const asked = lesson.asked.split(" ").map((word, i) => [word, i * ASK_STEP]);
  const pre = asked.length * ASK_STEP + GAP;
  const askBack = [];
  let next = pre + lesson.askAt + SWAP;
  for (const word of lesson.askBack.split(" ")) {
    askBack.push([word, next]);
    next += word.endsWith(",") ? 0.32 : 0.2;
  }
  return {
    pre,
    asked,
    askBack,
    listenAt: askBack.at(-1)[1] + LISTEN,
    total: pre + lesson.length,
  };
}

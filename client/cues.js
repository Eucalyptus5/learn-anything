export function createCues({ setTimer, clearTimer, send, apply, version }) {
  const held = new Map();
  let epoch = 0;
  let barrier = 0;
  let revision = 0;
  let sceneId = null;
  let step = 0;
  let lastCue = 0;
  let fired = 0;
  let refusing = false;

  function position() {
    return { scene_id: sceneId, step, revision };
  }

  function ack(cue, outcome, reason) {
    send({
      type: "lesson.ack", epoch: cue.epoch, barrier: cue.barrier, cue_id: cue.cue_id,
      outcome, reason, ...position(),
    });
  }

  function mismatch(cue) {
    if (cue.epoch !== epoch) return "stale_epoch";
    if (cue.barrier !== barrier) return "stale_barrier";
    if (refusing || cue.revision !== revision || cue.scene_id !== sceneId) return "stale_revision";
    if (cue.tag.kind === "step" && cue.tag.n <= step) return "range";
    return null;
  }

  function fire(cue) {
    const reason = mismatch(cue);
    if (reason !== null) {
      ack(cue, "dropped", reason);
      return;
    }
    if (cue.tag.kind === "scene") {
      sceneId = cue.tag.scene_id;
      step = 1;
    } else {
      step = cue.tag.n;
    }
    revision += 1;
    lastCue = cue.cue_id;
    fired += 1;
    apply({ scene_id: sceneId, step, tag: cue.tag });
    ack(cue, "fired", null);
  }

  function due(cueId) {
    const ready = [...held.keys()].filter((id) => id <= cueId).sort((a, b) => a - b);
    for (const id of ready) {
      const record = held.get(id);
      held.delete(id);
      clearTimer(record.timer);
      fire(record.cue);
    }
    if (held.size === 0 && fired > 0) {
      fired = 0;
      send({ type: "lesson.checkpoint", epoch, scene_id: sceneId, version: version(sceneId), step, revision });
    }
  }

  function drop(reason) {
    const ids = [...held.keys()].sort((a, b) => a - b);
    for (const id of ids) {
      const record = held.get(id);
      held.delete(id);
      clearTimer(record.timer);
      ack(record.cue, "dropped", reason);
    }
  }

  return {
    attach(next) {
      drop("barrier");
      epoch = next;
      barrier = 0;
      revision = 0;
      sceneId = null;
      step = 0;
      lastCue = 0;
      fired = 0;
      refusing = false;
    },
    hold(cue) {
      held.set(cue.cue_id, { cue, timer: setTimer(() => due(cue.cue_id), cue.lead_ms) });
    },
    sync(message) {
      if (message.epoch !== epoch) return;
      drop("barrier");
      barrier = message.barrier;
      refusing = false;
      send({ type: "lesson.synced", epoch, barrier, ...position(), last_cue: lastCue });
    },
    restore(restored) {
      sceneId = restored.scene_id;
      step = restored.step;
      revision = restored.revision;
      refusing = true;
      send({ type: "lesson.checkpoint", epoch, scene_id: sceneId, version: version(sceneId), step, revision });
    },
    drop,
    position,
    held: () => [...held.keys()],
  };
}

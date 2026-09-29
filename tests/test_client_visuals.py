import json
import re
import subprocess

from tutor.signaling import CLIENT_ROOT

CLIENT = CLIENT_ROOT / "client.js"
VISUALS = CLIENT_ROOT / "visuals.js"
FRAME = CLIENT_ROOT / "frame.html"
LESSON = CLIENT_ROOT / "lesson.js"
INDEX = CLIENT_ROOT / "index.html"
VISUAL_CHECK = CLIENT_ROOT / "visual_check.html"
SCENE_REVIEW = CLIENT_ROOT / "scene_review.html"
CUES = CLIENT_ROOT / "cues.js"
GUARDED = [CLIENT, VISUALS, FRAME, LESSON, CUES]
HOST_POLICY = {
    "default-src": ["'none'"],
    "script-src": ["'self'", "'unsafe-inline'"],
    "style-src": ["'self'", "'unsafe-inline'"],
    "connect-src": ["'self'"],
    "img-src": ["'self'", "data:"],
    "font-src": ["'self'"],
    "frame-src": ["'self'"],
    "base-uri": ["'none'"],
    "form-action": ["'none'"],
}
POLICY_META = re.compile(r'<meta http-equiv="Content-Security-Policy" content="([^"]*)">')
HTML_SINKS = [
    re.compile(r"insertAdjacentHTML\s*\("),
    re.compile(r"\.outerHTML\s*="),
    re.compile(r"document\.write\s*\("),
    re.compile(r"\.innerHTML\s*="),
]


def host_policy(text: str) -> dict[str, list[str]]:
    metas = POLICY_META.findall(text)
    assert len(metas) == 1
    directives = [directive.split() for directive in metas[0].split(";") if directive.strip()]
    assert len(directives) == len(HOST_POLICY)
    return {name: sources for name, *sources in directives}


def test_sandbox_attribute_is_exactly_allow_scripts() -> None:
    texts = {path.name: path.read_text() for path in GUARDED}
    sandboxes = re.findall(r'setAttribute\("sandbox",\s*"([^"]*)"\)', texts["visuals.js"])
    assert sandboxes == ["allow-scripts"]
    assert len(re.findall(r'createElement\("iframe"\)', texts["visuals.js"])) == 1
    assert 'createElement("iframe")' not in texts["client.js"]
    for name, text in texts.items():
        assert ".sandbox.add(" not in text, name
        assert ".sandbox =" not in text, name


def test_no_eval_or_function_constructor_in_client_sources() -> None:
    for path in GUARDED:
        assert not re.search(r"\b(new Function|eval)\s*\(", path.read_text()), path.name


def test_frame_creation_never_sets_allow_same_origin() -> None:
    for path in GUARDED:
        assert "allow-same-origin" not in path.read_text(), path.name


def test_no_innerhtml_assignment() -> None:
    for path in (CLIENT, VISUALS, LESSON):
        assert not re.search(r"\.innerHTML\s*=", path.read_text()), path.name


def test_the_offer_body_carries_the_session_from_the_card() -> None:
    text = CLIENT.read_text()
    session = re.search(r"session: \{(.*?)\}", text, re.DOTALL)
    assert session is not None
    for key in ("subject", "folder", "starting_from"):
        assert re.search(rf"\b{key}:", session[1]), key


def test_the_theme_is_sent_on_open_and_on_change() -> None:
    text = CLIENT.read_text()
    assert 'matchMedia("(prefers-color-scheme: dark)")' in text
    senders = [
        name
        for name, body in re.findall(r"function (\w+)\(\) \{(.*?)\n\}", text, re.DOTALL)
        if 'sendJson({ type: "theme"' in body
    ]
    assert len(senders) == 1
    assert f"onOpen({senders[0]})" in text
    change = re.search(r'addEventListener\("change",\s*\(\)\s*=>\s*\{(.*?)\n\}\);', text, re.DOTALL)
    assert change is not None
    assert f"{senders[0]}()" in change[1]


def test_captions_and_transcript_use_text_content_only() -> None:
    for path in (CLIENT, VISUALS, LESSON, CUES):
        for sink in HTML_SINKS:
            assert not sink.search(path.read_text()), (path.name, sink.pattern)
    client = CLIENT.read_text()
    for kind in ("caption", "transcript"):
        handler = re.search(
            rf'onPayload\("{kind}", \(payload\) => \{{(.*?)\n\}}\);', client, re.DOTALL
        )
        assert handler is not None, kind
        assert "textContent" in handler[1], kind
        for sink in HTML_SINKS:
            assert not sink.search(handler[1]), (kind, sink.pattern)


def test_history_entries_are_rebuilt_from_stored_payloads() -> None:
    visuals = VISUALS.read_text()
    assert "history.push(" in visuals
    receive = re.search(r"export function receive\(payload\) \{(.*?)\n\}", visuals, re.DOTALL)
    show = re.search(r"export function show\(i\) \{(.*?)\n\}", visuals, re.DOTALL)
    assert receive is not None and show is not None
    assert "render(payload)" in receive[1]
    assert "render(history[i].payload)" in show[1]
    client = CLIENT.read_text()
    assert re.search(r'addEventListener\("click",\s*\(\)\s*=>\s*show\(\w+\)\)', client)


def test_a_clear_and_a_reset_drop_the_live_app_frame() -> None:
    text = VISUALS.read_text()
    unmount = re.search(r"function unmountApp\(\) \{(.*?)\n\}", text, re.DOTALL)
    receive = re.search(r"export function receive\(payload\) \{(.*?)\n\}", text, re.DOTALL)
    reset = re.search(r"export function reset\(\) \{(.*?)\n\}", text, re.DOTALL)
    assert unmount is not None and receive is not None and reset is not None
    clear = re.search(r'case "diagram.clear":(.*?)break;', receive[1], re.DOTALL)
    assert clear is not None
    assert "app = null" in unmount[1]
    assert "frame.hidden = false" in unmount[1]
    for body in (clear[1], reset[1]):
        assert "unmountApp()" in body
        assert "clear: true" in body
        assert "announce(-1)" in body


def test_a_second_connect_reopens_the_channel_for_seq_reset() -> None:
    text = CLIENT.read_text()
    assert "export function onOpen" in text
    open_listener = re.search(
        r'channel\.addEventListener\("open",\s*\(\)\s*=>\s*\{(.*?)\n  \}\);', text, re.DOTALL
    )
    assert open_listener is not None
    assert re.search(r"openHandlers\.forEach\(|for \(const \w+ of openHandlers\)", open_listener[1])


def test_the_host_policy_is_exactly_the_pinned_directive_set() -> None:
    for path in (INDEX, VISUAL_CHECK, SCENE_REVIEW):
        assert host_policy(path.read_text()) == HOST_POLICY, path.name


def test_the_host_policy_precedes_every_script_and_style() -> None:
    for path in (INDEX, VISUAL_CHECK, SCENE_REVIEW):
        text = path.read_text()
        at = text.index('http-equiv="Content-Security-Policy"')
        assert at < text.index("<style"), path.name
        assert at < text.index("<script"), path.name


def test_the_validator_names_every_channel_payload_type() -> None:
    text = VISUALS.read_text()
    keys = re.search(r"const KEYS = \{(.*?)\n\};", text, re.DOTALL)
    assert keys is not None
    for kind in ("diagram.push", "diagram.clear", "source.highlight", "app.push"):
        assert re.search(rf'"{re.escape(kind)}": \[', keys[1]), kind
    for kind in ("state", "caption", "transcript"):
        assert re.search(rf'"{kind}": \["type", "seq", ', keys[1]), kind
    assert '"diagram.push": ["type", "seq", "id", "kind", "source", "title"]' in keys[1]
    assert '"app.push": ["type", "seq", "id", "html", "title"]' in keys[1]
    assert '"state": ["type", "seq", "state", "interrupted"]' in keys[1]
    assert '"caption": ["type", "seq", "turn_id", "text", "lead_ms"]' in keys[1]
    assert '"visual.pending"' not in keys[1]
    assert '"scene.push": ["type", "seq", "scene_id", "title", "html", "steps"]' in keys[1]
    assert '"scene.show": ["type", "seq", "scene_id", "at"]' in keys[1]
    assert '"scene.step": ["type", "seq", "scene_id", "n", "lead_ms"]' in keys[1]


def test_the_validator_checks_the_lead_and_the_interrupted_flag() -> None:
    text = VISUALS.read_text()
    validate = re.search(r"export function validate\(payload\) \{(.*?)\n\}", text, re.DOTALL)
    assert validate is not None
    assert "Number.isInteger(payload.lead_ms)" in validate[1]
    assert "payload.lead_ms < 0" in validate[1]
    assert 'typeof payload.interrupted !== "boolean"' in validate[1]


def test_the_check_page_probes_the_lead_and_the_interrupted_flag() -> None:
    text = VISUAL_CHECK.read_text()
    good = re.search(r"const good = \[(.*?)\n\];", text, re.DOTALL)
    hostile = re.search(r"const hostile = \[(.*?)\n\];", text, re.DOTALL)
    assert good is not None and hostile is not None
    assert "lead_ms: 1200" in good[1]
    assert "interrupted: false" in good[1] and "interrupted: true" in good[1]
    for name in (
        '"lead_ms -1"',
        '"lead_ms 1.5"',
        '"caption missing lead_ms"',
        '"interrupted yes"',
        '"state missing interrupted"',
    ):
        assert name in hostile[1], name


def test_no_phase_and_no_pending_title_reach_the_page() -> None:
    for path in (CLIENT, VISUALS, INDEX, VISUAL_CHECK):
        text = path.read_text()
        assert "phase" not in text.lower(), path.name
        assert "visual.pending" not in text, path.name
    assert ".canvas.drawing" not in INDEX.read_text()
    assert 'classList.add("drawing")' not in CLIENT.read_text()


def test_the_caption_handler_holds_each_clause_for_its_lead() -> None:
    client = CLIENT.read_text()
    handler = re.search(r'onPayload\("caption", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL)
    assert handler is not None
    assert "setTimeout(" in handler[1]
    assert "payload.lead_ms)" in handler[1]
    assert "startTurn(payload.turn_id)" in handler[1]
    assert "said.scrollTop = said.scrollHeight" in handler[1]


def test_held_captions_drop_only_on_an_interrupted_state() -> None:
    client = CLIENT.read_text()
    state = re.search(r'onPayload\("state", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL)
    assert state is not None
    assert re.search(r"if \(payload\.interrupted\) dropHeld\(\);", state[1])
    assert state[1].count("dropHeld()") == 1
    transcript = re.search(
        r'onPayload\("transcript", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL
    )
    assert transcript is not None
    assert "startTurn(payload.turn_id)" in transcript[1]


def test_a_natural_listening_state_waits_for_the_held_captions() -> None:
    client = CLIENT.read_text()
    state = re.search(r'onPayload\("state", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL)
    caption = re.search(r'onPayload\("caption", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL)
    drop = re.search(r"function dropHeld\(\) \{(.*?)\n\}", client, re.DOTALL)
    assert state is not None and caption is not None and drop is not None
    assert 'payload.state === "listening" && !payload.interrupted && held.size > 0' in state[1]
    assert "pendingState = payload" in state[1]
    stash = state[1].index("pendingState = payload")
    assert state[1].index("if (payload.interrupted) dropHeld();") < stash
    assert state[1].count("applyState(payload)") == 1
    assert "liveText.textContent" not in state[1]
    assert "held.size === 0 && pendingState !== null" in caption[1]
    assert "applyState(pendingState)" in caption[1]
    assert "pendingState = null" in drop[1]


def test_the_reply_card_replaces_the_caption_line() -> None:
    index = INDEX.read_text()
    client = CLIENT.read_text()
    assert 'class="reply"' in index and 'class="you"' in index and 'class="said"' in index
    assert 'class="caption"' not in index
    assert ".caption" not in index and ".caption" not in client
    assert "max-height: 4.5em" in index
    assert ".canvas iframe.landing" in index


def test_the_reduced_motion_block_closes_the_stylesheet() -> None:
    index = INDEX.read_text()
    sheet = re.search(r"<style>(.*?)</style>", index, re.DOTALL)
    assert sheet is not None
    reduced = re.search(
        r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n  \}", sheet[1], re.DOTALL
    )
    assert reduced is not None
    assert ".canvas iframe { transition: none; }" in reduced[1]
    assert ".canvas iframe.landing { opacity: 1; transform: none; }" in reduced[1]
    assert sheet[1][reduced.end() :].strip() == ""
    for selector in (".canvas iframe {", ".canvas iframe.landing {"):
        assert sheet[1].index(selector) < reduced.start(), selector


def test_a_new_frame_lands_with_the_landing_class_until_it_loads() -> None:
    text = VISUALS.read_text()
    frame = re.search(r"function sandboxedFrame\(\) \{(.*?)\n\}", text, re.DOTALL)
    assert frame is not None
    assert 'classList.add("landing")' in frame[1]
    assert re.search(
        r'addEventListener\("load", \(\) => element\.classList\.remove\("landing"\)', frame[1]
    )


def test_every_push_on_the_check_page_carries_a_title() -> None:
    text = VISUAL_CHECK.read_text()
    good = re.search(r"const good = \[(.*?)\n\];", text, re.DOTALL)
    hostile = re.search(r"const hostile = \[(.*?)\n\];", text, re.DOTALL)
    assert good is not None and hostile is not None
    pushes = re.findall(r'\{ type: "(?:diagram|app)\.push"[^}]*\}', good[1])
    receives = re.findall(r'receive\(\{ type: "(?:diagram|app)\.push"[^}]*\}\)', text)
    assert len(pushes) == 2
    assert len(receives) == 4
    for literal in pushes + receives:
        assert "title:" in literal, literal
    for name in ('"title of 81"', '"app.push missing title"', '"diagram.push missing title"'):
        assert name in hostile[1], name


def test_the_lesson_helper_reaches_nothing_outside_its_frame_but_its_one_report() -> None:
    text = LESSON.read_text()
    for pattern in (
        r"\btop\.",
        r"fetch\(",
        r"XMLHttpRequest",
        r"WebSocket",
        r"import\(",
        r"document\.cookie",
        r"localStorage",
        r"\.srcdoc",
        r"opener",
    ):
        assert not re.search(pattern, text), pattern
    assert text.count("postMessage(") == 1
    assert len(re.findall(r"\bparent\b", text)) == 1
    report = re.search(r"function report\(steps, root\) \{(.*?)\n  \}", text, re.DOTALL)
    assert report is not None
    assert "if (reported) return;" in report[1]
    assert (
        'window.parent.postMessage(\n      { type: "scene.ready", steps, width: box.width, '
        'height: box.height, error },\n      "*",\n    );'
    ) in report[1]
    assert "window.lesson = Object.freeze({ scene })" in text
    assert "grid-template-columns: 1fr 220px" in text
    assert "prefers-reduced-motion" in text
    assert "REPORT_MS = 5000" in text
    assert "error.slice(0, 500)" in text or "String(message).slice(0, 500)" in text
    assert "setTimeout" in text and "requestAnimationFrame" in text
    for gone in ("FIRST_MS", "BEAT_MS", "BEAT_PER_CHAR_MS", "BEAT_CAP_MS", "FADE_MS", "DRAW_MS"):
        assert gone not in text, gone
    assert "lesson.steps" not in text and "function steps(" not in text
    assert len(text.splitlines()) < 170


def test_the_lesson_helper_takes_only_an_integer_step_in_range() -> None:
    text = LESSON.read_text()
    listener = re.search(
        r'window\.addEventListener\("message", \(event\) => \{(.*?)\n  \}\);', text, re.DOTALL
    )
    assert listener is not None
    assert "Number.isInteger(m.step)" in listener[1]
    assert "m.step < 1 || m.step > current.steps.length" in listener[1]
    assert "go(m.step)" in listener[1]
    assert "event.source" not in listener[1] and "origin" not in listener[1]
    assert text.count('addEventListener("message"') == 1


def test_the_lesson_helper_demands_a_label_per_step() -> None:
    text = LESSON.read_text()
    register = re.search(r"function register\(spec\) \{(.*?)\n  \}", text, re.DOTALL)
    assert register is not None
    assert 'throw new Error("lesson.scene was called twice")' in register[1]
    assert 'typeof timeline.tweenTo !== "function"' in register[1]
    assert "timeline.labels" in register[1]
    assert "timeline.pause().seek(label(1))" in register[1]
    assert "mark(1)" in register[1]


def test_the_lesson_helper_never_reports_while_the_page_is_hidden() -> None:
    text = LESSON.read_text()
    fallback = re.search(r"function fallback\(\) \{(.*?)\n  \}", text, re.DOTALL)
    assert fallback is not None
    assert 'if (document.visibilityState === "hidden") {' in fallback[1]
    assert (
        'document.addEventListener("visibilitychange", () => setTimeout(fallback, REPORT_MS), '
        "{ once: true });"
    ) in fallback[1]
    assert fallback[1].index('visibilityState === "hidden"') < fallback[1].index("report(")
    assert "setTimeout(fallback, REPORT_MS);" in text
    assert text.count("visibilitychange") == 1


def test_the_check_page_narrates_one_scene_through_the_helper() -> None:
    text = VISUAL_CHECK.read_text()
    scene = re.search(r"const sceneHtml = `(.*?)`;", text, re.DOTALL)
    assert scene is not None
    assert '<script src="/lesson.js"><\\/script>' in scene[1]
    assert '<script src="/vendor/gsap.min.js"><\\/script>' in scene[1]
    assert "gsap.timeline({ paused: true })" in scene[1]
    for n in (1, 2, 3):
        assert f'addLabel("step-{n}")' in scene[1]
    assert "lesson.scene({" in scene[1]
    assert '"scene: ok"' in text


def test_the_check_page_loads_gsap_in_the_sandbox() -> None:
    text = VISUAL_CHECK.read_text()
    libs = re.search(r"const libsHtml = `(.*?)`;", text, re.DOTALL)
    assert libs is not None
    assert '<script src="/vendor/gsap.min.js"><\\/script>' in libs[1]
    assert 'typeof gsap.timeline === "function"' in libs[1]
    assert 'log("gsap: " + gsap)' in text
    assert 'katex && plotly && gsap ? "libs: ok"' in text


def say_handler(text: str) -> str:
    handler = re.search(
        r'say\.addEventListener\("keydown",\s*\(event\)\s*=>\s*\{(.*?)\n\}\);', text, re.DOTALL
    )
    assert handler is not None
    return handler[1]


def test_the_typed_line_is_sent_as_a_say_and_cleared() -> None:
    client = CLIENT.read_text()
    assert 'getElementById("say")' in client
    handler = say_handler(client)
    assert 'sendJson({ type: "say", text' in handler
    assert '.value = ""' in handler
    assert 'readyState === "open"' in handler
    index = INDEX.read_text()
    assert '<input id="say"' in index
    assert 'placeholder="type instead of speaking"' in index
    assert 'maxlength="4000"' in index


def test_the_typed_line_submits_on_enter_only() -> None:
    handler = say_handler(CLIENT.read_text())
    assert 'event.key === "Enter"' in handler
    assert 'sendJson({ type: "say"' not in handler.split('event.key === "Enter"')[0]
    assert CLIENT.read_text().count("say.addEventListener(") == 1


def export_body(text: str) -> str:
    body = re.search(r"function exportSession\(format\) \{(.*?)\nfunction ", text, re.DOTALL)
    assert body is not None
    return body[1]


def keydown_handler(text: str) -> str:
    handler = re.search(
        r'document\.addEventListener\("keydown",\s*\(event\)\s*=>\s*\{(.*?)\n\}\);',
        text,
        re.DOTALL,
    )
    assert handler is not None
    return handler[1]


def test_each_export_key_downloads_one_file() -> None:
    client = CLIENT.read_text()
    body = export_body(client)
    assert body.count("download(") == 2
    json_branch, markdown_branch = body.split("} else {")
    assert 'if (format === "json")' in json_branch
    assert json_branch.count("download(") == 1
    assert '"application/json"' in json_branch
    assert '"text/markdown"' not in json_branch
    assert markdown_branch.count("download(") == 1
    assert '"text/markdown"' in markdown_branch
    assert '"application/json"' not in markdown_branch
    assert client.count(".download =") == 1
    assert "URL.createObjectURL(" in client
    assert "URL.revokeObjectURL(" in client
    handler = keydown_handler(client)
    e_branch = re.search(r'event\.key === "e"(.*?)else if', handler, re.DOTALL)
    m_branch = re.search(r'event\.key === "m"(.*?)$', handler, re.DOTALL)
    assert e_branch is not None and m_branch is not None
    assert 'exportSession("json")' in e_branch[1]
    assert 'exportSession("markdown")' not in e_branch[1]
    assert 'exportSession("markdown")' in m_branch[1]
    assert 'exportSession("json")' not in m_branch[1]
    assert handler.count("exportSession(") == 2
    assert 'const open = channel !== null && channel.readyState === "open";' in handler
    assert 'event.key === "e" && open' in handler
    assert 'event.key === "m" && open' in handler
    begin = re.search(r"function begin\(\) \{(.*?)\n\}", client, re.DOTALL)
    assert begin is not None
    assert "turns.length = 0" in begin[1]
    index = INDEX.read_text()
    keys = re.search(r'<p class="keys">(.*?)</p>', index)
    assert keys is not None
    for key in ("t", "d", "e", "m"):
        assert f"<span><kbd>{key}</kbd>" in keys[1], key


def test_export_never_renders_a_visual() -> None:
    client = CLIENT.read_text()
    span = re.search(
        r"function download\(.*?function exportSession\(format\) \{.*?\n\}", client, re.DOTALL
    )
    assert span is not None
    assert "function markdown()" in span[0]
    assert "srcdoc" not in span[0]
    assert 'createElement("iframe")' not in span[0]
    assert "innerHTML" not in span[0]
    assert "entries()" in export_body(client)
    assert "export function entries()" in VISUALS.read_text()


def test_the_markdown_lists_visuals_by_seq_after_the_turns_not_by_caption_range() -> None:
    client = CLIENT.read_text()
    builder = re.search(r"function markdown\(\) \{(.*?)\n\}", client, re.DOTALL)
    assert builder is not None
    assert "caption" not in builder[1]
    assert "seq" not in builder[1]
    assert builder[1].index("turns") < builder[1].index("## visuals")
    assert builder[1].index("## visuals") < builder[1].index("entries()")
    assert "_visual: ${" in builder[1]


def test_the_frame_answers_before_the_staged_reveal_and_restarts_it_per_render() -> None:
    text = FRAME.read_text()
    render = re.search(r"async function render\(m\) \{(.*?)\n\}", text, re.DOTALL)
    clear = re.search(r"async function clear\(\) \{(.*?)\n\}", text, re.DOTALL)
    stage = re.search(r"function stage\(svg, kind, token\) \{(.*?)\n\}", text, re.DOTALL)
    assert render is not None and clear is not None and stage is not None
    assert "reveal += 1" in render[1] and "reveal += 1" in clear[1]
    assert "postMessage" not in render[1] and "postMessage" not in stage[1]
    assert "setTimeout(" in stage[1] and "token !== reveal" in stage[1]
    assert "source.postMessage({ seq: m.seq, ok }, origin)" in text
    assert "prefers-reduced-motion" in text
    assert "transform" not in text


def test_edge_ends_resolve_under_the_render_id_prefix_mermaid_writes() -> None:
    text = FRAME.read_text()
    functions = re.findall(
        r"^function (?:nodeName|edgeId|edgeEnds)\(.*?\n\}", text, re.MULTILINE | re.DOTALL
    )
    assert len(functions) == 3
    probe = (
        "\n".join(functions)
        + """
const ids = ["flowchart-a-0", "flowchart-b-1", "flowchart-x_1-2", "flowchart-y_2-3"];
const names = new Set(ids.map(nodeName));
process.stdout.write(JSON.stringify({
  a_b: edgeEnds(edgeId("m1-L_a_b_0"), names),
  a_b_10: edgeEnds(edgeId("m12-L_a_b_10"), names),
  underscored: edgeEnds(edgeId("m1-L_x_1_y_2_0"), names),
  bare: edgeEnds(edgeId("L_b_a_0"), names),
  label: edgeId("m1-L_a_b_0"),
}));
"""
    )
    run = subprocess.run(["node", "-e", probe], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == {
        "a_b": ["a", "b"],
        "a_b_10": ["a", "b"],
        "underscored": ["x_1", "y_2"],
        "bare": ["b", "a"],
        "label": "L_a_b_0",
    }
    flowchart = re.search(r"function flowchartSteps\(svg\) \{(.*?)\n\}", text, re.DOTALL)
    assert flowchart is not None
    assert "edgeEnds(edgeId(path.id), names)" in flowchart[1]
    assert "labels.get(edgeId(path.id))" in flowchart[1]


def test_a_scene_is_checked_in_a_hidden_frame_and_shown_only_for_its_id() -> None:
    visuals = VISUALS.read_text()
    assert len(re.findall(r'createElement\("iframe"\)', visuals)) == 1
    check = re.search(r"function check\(payload\) \{(.*?)\n\}", visuals, re.DOTALL)
    promote = re.search(r"function promote\(at\) \{(.*?)\n\}", visuals, re.DOTALL)
    receive = re.search(r"export function receive\(payload\) \{(.*?)\n\}", visuals, re.DOTALL)
    reset = re.search(r"export function reset\(\) \{(.*?)\n\}", visuals, re.DOTALL)
    assert check is not None and promote is not None and receive is not None and reset is not None
    assert "dropChecking()" in check[1]
    assert 'classList.add("checking")' in check[1]
    assert "checking = { payload, frame: element }" in check[1]
    assert 'classList.add("landing")' in promote[1]
    assert 'classList.remove("checking")' in promote[1]
    assert 'requestAnimationFrame(() => element.classList.remove("landing"))' in promote[1]
    assert "mountApp(element)" in promote[1]
    assert "history.push({ payload, title: payload.title })" in promote[1]
    show = re.search(r'case "scene\.show":(.*?)case "scene\.step":', receive[1], re.DOTALL)
    assert show is not None
    assert "checking.payload.scene_id !== payload.scene_id" in show[1]
    assert "promote(payload.at)" in show[1]
    assert re.search(r'case "scene\.push":\s*check\(payload\);', receive[1])
    assert re.search(r'case "scene\.step":\s*listeners\.get\("step"\)\?\.\(payload\);', receive[1])
    assert "dropChecking()" in reset[1]
    index = INDEX.read_text()
    assert (
        ".canvas iframe.checking { opacity: 0; pointer-events: none; transform: translateY(10px); }"
        in index
    )
    assert ".canvas iframe.leaving" in index
    assert "grid-template-rows: auto minmax(0, 1fr) auto" in index


def test_the_ready_report_is_taken_from_the_checking_frame_only_and_compared_to_the_push() -> None:
    visuals = VISUALS.read_text()
    listener = re.search(
        r'window\.addEventListener\("message", \(event\) => \{(.*?)\n\}\);', visuals, re.DOTALL
    )
    assert listener is not None
    assert "checking === null || event.source !== checking.frame.contentWindow" in listener[1]
    assert 'm.type !== "scene.ready"' in listener[1]
    assert "Number.isInteger(m.steps)" in listener[1]
    assert "Number.isFinite(m.width)" in listener[1] and "Number.isFinite(m.height)" in listener[1]
    assert 'typeof m.error !== "string"' in listener[1]
    assert "m.error.slice(0, ERROR_CAP)" in listener[1]
    assert "m.steps === payload.steps.length" in listener[1]
    assert "m.width > 0 && m.height > 0" in listener[1]
    assert '"reported " + m.steps + " steps, pushed " + payload.steps.length' in listener[1]
    assert '"root has no size " + Math.round(m.width) + "x" + Math.round(m.height)' in listener[1]
    assert 'const ok = error === "";' in listener[1]
    assert "if (checking.reported) return;\n  checking.reported = true;" in listener[1]
    assert "if (!ok) dropChecking();" in listener[1]
    dropped = listener[1].index("if (!ok) dropChecking();")
    assert dropped < listener[1].index('listeners.get("ready")?.(')
    assert 'listeners.get("ready")?.(' in listener[1]
    assert "const ERROR_CAP = 500;" in visuals
    assert visuals.count('addEventListener("message"') == 1
    for sink in HTML_SINKS:
        assert not sink.search(listener[1]), sink.pattern


def test_a_step_is_held_for_its_lead_and_reaches_the_frame_only_through_step_scene() -> None:
    client = CLIENT.read_text()
    handler = re.search(r'onPayload\("step", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL)
    assert handler is not None
    assert "setTimeout(" in handler[1]
    assert "payload.lead_ms)" in handler[1]
    assert "held.add(timer)" in handler[1]
    assert "stepScene(payload.n)" in handler[1]
    assert "postMessage" not in client
    for kind in ("scene.push", "scene.show", "scene.step"):
        assert f'onJson("{kind}", receive)' in client, kind
    assert "postMessage" not in CUES.read_text()
    ready = re.search(r'onPayload\("ready", \(report\) => \{(.*?)\n\}\);', client, re.DOTALL)
    assert ready is not None
    assert 'sendJson({ type: "scene.ready", ...report })' in ready[1]
    assert 'channel.readyState === "open"' in ready[1]
    visuals = VISUALS.read_text()
    step = re.search(r"export function stepScene\(n\) \{(.*?)\n\}", visuals, re.DOTALL)
    assert step is not None
    assert 'target.contentWindow.postMessage({ step: n }, "*")' in step[1]
    assert "if (target === app)" in step[1]
    assert visuals.count("postMessage(") == 2
    assert 'frame.contentWindow.postMessage(message, "*")' in visuals


def test_the_validator_checks_the_scene_id_the_steps_and_the_counts() -> None:
    visuals = VISUALS.read_text()
    validate = re.search(r"export function validate\(payload\) \{(.*?)\n\}", visuals, re.DOTALL)
    assert validate is not None
    assert "const SCENE_ID = /^[a-z][a-z0-9-]{0,31}$/;" in visuals
    push = re.search(r'case "scene\.push":(.*?)return true;', validate[1], re.DOTALL)
    show = re.search(r'case "scene\.show":(.*?)return true;', validate[1], re.DOTALL)
    step = re.search(r'case "scene\.step":(.*?)return true;', validate[1], re.DOTALL)
    assert push is not None and show is not None and step is not None
    assert "sceneId(payload.scene_id)" in push[1] and "stepList(payload.steps)" in push[1]
    assert 'cappedString(payload, "html", SCENE_HTML_CAP)' in push[1]
    assert 'payload.html === ""' in push[1]
    assert "positiveInteger(payload.at)" in show[1]
    assert "positiveInteger(payload.n)" in step[1]
    assert "payload.lead_ms < 0" in step[1]
    assert "const SCENE_HTML_CAP = 200000;" in visuals
    assert "const STEP_CAP = 120;" in visuals and "const STEPS_MAX = 8;" in visuals


def test_the_check_page_probes_the_scene_payloads_and_a_broken_scene() -> None:
    text = VISUAL_CHECK.read_text()
    good = re.search(r"const good = \[(.*?)\n\];", text, re.DOTALL)
    hostile = re.search(r"const hostile = \[(.*?)\n\];", text, re.DOTALL)
    assert good is not None and hostile is not None
    for name in ('"scene.push"', '"scene.show"', '"scene.step"'):
        assert name in good[1], name
    for name in (
        '"scene.push scene_id Bad"',
        '"scene.push html of 200001"',
        '"scene.push empty html"',
        '"scene.push nine steps"',
        '"scene.push step of 121"',
        '"scene.push no steps"',
        '"scene.show at 0"',
        '"scene.step n 0"',
        '"scene.step lead_ms -1"',
    ):
        assert name in hostile[1], name
    assert len(re.findall(r'receive\(\{ type: "scene\.push"', text)) == 2
    assert 'receive({ type: "scene.show", seq: 105, scene_id: "check-scene", at: 1 })' in text
    assert "stepScene(2)" in text and "stepScene(3)" in text
    assert 'onPayload("ready", (report) => {' in text
    assert '"scene: ok"' in text and '"broken scene: reported"' in text
    assert "const brokenHtml" in text
    assert re.search(r"#log \{[^}]*max-height: 10rem; overflow: auto; \}", text)


def test_the_review_page_reads_files_locally_and_checks_them_through_the_host() -> None:
    text = SCENE_REVIEW.read_text()
    assert (
        'import { mount, onPayload, receive, reset, show, stepScene } from "/visuals.js";' in text
    )
    assert '<input id="files" type="file" multiple accept=".html,.json">' in text
    assert "new FileReader()" in text or ".text()" in text
    for banned in ("fetch(", "XMLHttpRequest", "WebSocket", "srcdoc", "innerHTML", "eval("):
        assert banned not in text, banned
    assert 'receive({ type: "scene.push"' in text
    assert 'receive({ type: "scene.show"' in text
    assert 'onPayload("ready", (report) => {' in text
    assert 'onPayload("history", (items, currentIndex) => {' in text
    assert "stepScene(n)" in text
    assert "passed " in text and "of " in text
    assert "const REPORT_WAIT_MS = 20000;" in text
    advance = re.search(r"function next\(\) \{(.*?)\n\}", text, re.DOTALL)
    ready = re.search(r'onPayload\("ready", \(report\) => \{(.*?)\n\}\);', text, re.DOTALL)
    assert advance is not None and ready is not None
    assert "timer = setTimeout(" in advance[1] and "REPORT_WAIT_MS)" in advance[1]
    assert '": no report, skipped"' in advance[1] and "next();" in advance[1]
    assert "clearTimeout(timer)" in ready[1]
    change = re.search(
        r'addEventListener\("change", async \(event\) => \{(.*?)\n\}\);', text, re.DOTALL
    )
    assert change is not None
    assert "clearTimeout(timer)" in change[1] and "reset();" in change[1]


CUE_HARNESS = """
const timers = new Map();
let nextTimer = 0;
const sent = [];
const applied = [];
const cues = createCues({
  setTimer: (run, ms) => {
    nextTimer += 1;
    timers.set(nextTimer, { run, ms });
    return nextTimer;
  },
  clearTimer: (id) => timers.delete(id),
  send: (message) => sent.push(message),
  apply: (position) => applied.push(position),
  version: () => 0,
});
function elapse(id) {
  const timer = timers.get(id);
  timers.delete(id);
  timer.run();
}
function cue(fields) {
  return {
    type: "lesson.cue", seq: 1, epoch: 1, barrier: 0, chunk_id: 1, scene_id: null,
    revision: 0, lead_ms: 400, audio_ms: 900, ...fields,
  };
}
const OPEN = { kind: "scene", n: 1, scene_id: "ratio" };
"""


def run_cues(probe: str) -> dict[str, object]:
    report = (
        "\nconsole.log(JSON.stringify({ sent, applied, held: cues.held(), "
        "timers: [...timers.keys()], position: cues.position() }));"
    )
    source = CUES.read_text() + CUE_HARNESS + probe + report
    run = subprocess.run(
        ["node", "--input-type=module", "-e", source], capture_output=True, text=True, check=True
    )
    return json.loads(run.stdout)


def ack(
    cue_id: int,
    outcome: str,
    reason: str | None,
    scene: str | None,
    step: int,
    revision: int,
    barrier: int = 0,
) -> dict[str, object]:
    return {
        "type": "lesson.ack",
        "epoch": 1,
        "barrier": barrier,
        "cue_id": cue_id,
        "outcome": outcome,
        "reason": reason,
        "scene_id": scene,
        "step": step,
        "revision": revision,
    }


def test_a_cue_fires_at_its_lead_acks_its_position_and_the_settled_page_checkpoints() -> None:
    out = run_cues("""
cues.attach(1);
cues.hold(cue({ cue_id: 1, tag: OPEN }));
cues.hold(cue({ cue_id: 2, chunk_id: 2, scene_id: "ratio", revision: 1, tag: { kind: "step", n: 2 } }));
elapse(1);
elapse(2);
""")
    assert out["sent"] == [
        ack(1, "fired", None, "ratio", 1, 1),
        ack(2, "fired", None, "ratio", 2, 2),
        {
            "type": "lesson.checkpoint",
            "epoch": 1,
            "scene_id": "ratio",
            "version": 0,
            "step": 2,
            "revision": 2,
        },
    ]
    assert [(p["scene_id"], p["step"]) for p in out["applied"]] == [("ratio", 1), ("ratio", 2)]
    assert out["held"] == [] and out["timers"] == []


def test_a_later_timer_fires_the_earlier_held_cue_first() -> None:
    out = run_cues("""
cues.attach(1);
cues.hold(cue({ cue_id: 1, lead_ms: 900, tag: OPEN }));
cues.hold(cue({ cue_id: 2, lead_ms: 300, scene_id: "ratio", revision: 1, tag: { kind: "step", n: 2 } }));
elapse(2);
""")
    assert [(m["cue_id"], m["outcome"]) for m in out["sent"] if m["type"] == "lesson.ack"] == [
        (1, "fired"),
        (2, "fired"),
    ]
    assert out["timers"] == []


def test_a_cue_the_page_cannot_take_is_dropped_with_its_reason() -> None:
    out = run_cues("""
cues.attach(1);
cues.hold(cue({ cue_id: 1, epoch: 2, tag: OPEN }));
cues.hold(cue({ cue_id: 2, barrier: 1, tag: OPEN }));
cues.hold(cue({ cue_id: 3, revision: 4, tag: OPEN }));
cues.hold(cue({ cue_id: 4, tag: OPEN }));
cues.hold(cue({ cue_id: 5, scene_id: "clip", revision: 1, tag: { kind: "step", n: 2 } }));
cues.hold(cue({ cue_id: 6, scene_id: "ratio", revision: 1, tag: { kind: "step", n: 1 } }));
elapse(6);
""")
    acks = [
        (m["cue_id"], m["outcome"], m["reason"]) for m in out["sent"] if m["type"] == "lesson.ack"
    ]
    assert acks == [
        (1, "dropped", "stale_epoch"),
        (2, "dropped", "stale_barrier"),
        (3, "dropped", "stale_revision"),
        (4, "fired", None),
        (5, "dropped", "stale_revision"),
        (6, "dropped", "range"),
    ]
    assert out["position"] == {"scene_id": "ratio", "step": 1, "revision": 1}


def test_a_sync_drops_every_held_cue_then_reports_the_page_and_adopts_the_barrier() -> None:
    out = run_cues("""
cues.attach(1);
cues.hold(cue({ cue_id: 1, tag: OPEN }));
elapse(1);
cues.hold(cue({ cue_id: 2, scene_id: "ratio", revision: 1, tag: { kind: "step", n: 2 } }));
cues.hold(cue({ cue_id: 3, scene_id: "ratio", revision: 2, tag: { kind: "step", n: 3 } }));
cues.sync({ epoch: 2, barrier: 1 });
cues.sync({ epoch: 1, barrier: 1 });
cues.hold(cue({ cue_id: 4, scene_id: "ratio", revision: 1, tag: { kind: "step", n: 2 } }));
cues.hold(cue({ cue_id: 5, barrier: 1, scene_id: "ratio", revision: 1, tag: { kind: "step", n: 2 } }));
elapse(5);
""")
    assert [m["type"] for m in out["sent"]] == [
        "lesson.ack",
        "lesson.checkpoint",
        "lesson.ack",
        "lesson.ack",
        "lesson.synced",
        "lesson.ack",
        "lesson.ack",
        "lesson.checkpoint",
    ]
    assert out["sent"][2] == ack(2, "dropped", "barrier", "ratio", 1, 1)
    assert out["sent"][3] == ack(3, "dropped", "barrier", "ratio", 1, 1)
    assert out["sent"][4] == {
        "type": "lesson.synced",
        "epoch": 1,
        "barrier": 1,
        "scene_id": "ratio",
        "step": 1,
        "revision": 1,
        "last_cue": 1,
    }
    assert out["sent"][5] == ack(4, "dropped", "stale_barrier", "ratio", 1, 1)
    assert out["sent"][6] == ack(5, "fired", None, "ratio", 2, 2, barrier=1)
    assert out["timers"] == []


def test_a_drop_acks_every_held_cue_and_an_attach_starts_the_page_clean() -> None:
    out = run_cues("""
cues.attach(1);
cues.hold(cue({ cue_id: 1, tag: OPEN }));
elapse(1);
cues.hold(cue({ cue_id: 2, scene_id: "ratio", revision: 1, tag: { kind: "step", n: 2 } }));
cues.drop("barrier");
cues.attach(2);
""")
    assert out["sent"][-1] == ack(2, "dropped", "barrier", "ratio", 1, 1)
    assert out["held"] == [] and out["timers"] == []
    assert out["position"] == {"scene_id": None, "step": 0, "revision": 0}


def test_after_a_restore_every_cue_is_refused_until_the_next_sync() -> None:
    out = run_cues("""
cues.attach(1);
cues.hold(cue({ cue_id: 1, tag: OPEN }));
elapse(1);
cues.hold(cue({ cue_id: 2, scene_id: "ratio", revision: 1, tag: { kind: "step", n: 2 } }));
elapse(2);
cues.restore({ scene_id: "ratio", step: 1, revision: 2 });
cues.hold(cue({ cue_id: 3, scene_id: "ratio", revision: 2, tag: { kind: "step", n: 2 } }));
elapse(3);
cues.sync({ epoch: 1, barrier: 1 });
cues.hold(cue({ cue_id: 4, barrier: 1, scene_id: "ratio", revision: 2, tag: { kind: "step", n: 2 } }));
elapse(4);
""")
    assert [m["type"] for m in out["sent"]] == [
        "lesson.ack",
        "lesson.checkpoint",
        "lesson.ack",
        "lesson.checkpoint",
        "lesson.checkpoint",
        "lesson.ack",
        "lesson.synced",
        "lesson.ack",
        "lesson.checkpoint",
    ]
    assert out["sent"][4] == {
        "type": "lesson.checkpoint",
        "epoch": 1,
        "scene_id": "ratio",
        "version": 0,
        "step": 1,
        "revision": 2,
    }
    assert out["sent"][5] == ack(3, "dropped", "stale_revision", "ratio", 1, 2)
    assert out["sent"][6] == {
        "type": "lesson.synced",
        "epoch": 1,
        "barrier": 1,
        "scene_id": "ratio",
        "step": 1,
        "revision": 2,
        "last_cue": 2,
    }
    assert out["sent"][7] == ack(4, "fired", None, "ratio", 2, 3, barrier=1)
    assert out["timers"] == []


def test_every_lesson_payload_reaches_its_listener_through_all_three_layers() -> None:
    visuals = VISUALS.read_text()
    client = CLIENT.read_text()
    keys = re.search(r"const KEYS = \{(.*?)\n\};", visuals, re.DOTALL)
    validate = re.search(r"export function validate\(payload\) \{(.*?)\n\}", visuals, re.DOTALL)
    receive = re.search(r"export function receive\(payload\) \{(.*?)\n\}", visuals, re.DOTALL)
    assert keys is not None and validate is not None and receive is not None
    assert '"lesson.attach": ["type", "seq", "epoch"]' in keys[1]
    assert (
        '"lesson.cue": ["type", "seq", "epoch", "barrier", "cue_id", "chunk_id", "scene_id", '
        '"revision", "lead_ms", "audio_ms", "tag"]'
    ) in keys[1]
    assert '"lesson.sync": ["type", "seq", "epoch", "barrier"]' in keys[1]
    assert '"lesson.state": ["type", "seq", "scenes", "current"]' in keys[1]
    for kind, listener in (
        ("lesson.attach", "attach"),
        ("lesson.cue", "cue"),
        ("lesson.sync", "sync"),
        ("lesson.state", "lesson"),
    ):
        assert f'case "{kind}":' in validate[1], kind
        assert re.search(
            rf'case "{re.escape(kind)}":\s*listeners\.get\("{listener}"\)\?\.\(payload\);',
            receive[1],
        ), kind
        assert f'onJson("{kind}", receive)' in client, kind
    cue_case = re.search(r'case "lesson\.cue":(.*?)return true;', validate[1], re.DOTALL)
    assert cue_case is not None
    for check in (
        "cueTag(payload.tag)",
        "nonNegative(payload.barrier)",
        "nonNegative(payload.chunk_id)",
        "nonNegative(payload.lead_ms)",
        "nonNegative(payload.audio_ms)",
    ):
        assert check in cue_case[1], check


def test_the_page_holds_cues_in_the_scheduler_and_every_drop_trigger_acks() -> None:
    client = CLIENT.read_text()
    assert client.startswith(
        'import { blank, entries, land, mount, onPayload, receive, reset, show, stepScene, theme } from "/visuals.js";\n'
        'import { createCues } from "/cues.js";'
    )
    drop = re.search(r"function dropHeld\(\) \{(.*?)\n\}", client, re.DOTALL)
    assert drop is not None and 'cues.drop("barrier")' in drop[1]
    for name, call in (
        ("attach", "cues.attach(payload.epoch)"),
        ("cue", "cues.hold(payload)"),
        ("sync", "cues.sync(payload)"),
    ):
        handler = re.search(
            rf'onPayload\("{name}", \(payload\) => \{{(.*?)\n\}}\);', client, re.DOTALL
        )
        assert handler is not None and call in handler[1], name
    send = re.search(r"send: \(message\) => \{(.*?)\},", client, re.DOTALL)
    assert send is not None and 'channel.readyState === "open"' in send[1]
    cues = CUES.read_text()
    for banned in ("document", "window", "setTimeout", "postMessage", "innerHTML"):
        assert banned not in cues, banned


def test_a_checked_scene_lands_when_its_cue_fires_or_late_at_the_pages_step() -> None:
    visuals = VISUALS.read_text()
    client = CLIENT.read_text()
    land = re.search(r"export function land\(sceneId, at\) \{(.*?)\n\}", visuals, re.DOTALL)
    assert land is not None
    assert "checking.reported" in land[1]
    assert "checking.payload.scene_id !== sceneId" in land[1]
    assert "promote(at)" in land[1]
    show = re.search(
        r"function showPosition\(\{ scene_id, step, tag \}\) \{(.*?)\n\}", client, re.DOTALL
    )
    assert show is not None
    assert "land(scene_id, 1)" in show[1] and "blank()" in show[1]
    assert "stepScene(step)" in show[1]
    ready = re.search(r'onPayload\("ready", \(report\) => \{(.*?)\n\}\);', client, re.DOTALL)
    assert ready is not None
    assert "cues.position()" in ready[1] and "land(scene_id, step)" in ready[1]
    assert 'sendJson({ type: "scene.ready", ...report })' in ready[1]
    assert "version: (sceneId) =>" in client
    assert visuals.count("postMessage(") == 2


def test_the_lesson_list_sits_above_the_visuals_strip_and_is_written_as_text() -> None:
    index = INDEX.read_text()
    client = CLIENT.read_text()
    assert index.index('<ol class="lesson">') < index.index("<h3>Visuals</h3>")
    lesson = re.search(r'onPayload\("lesson", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL)
    assert lesson is not None
    assert "item.textContent = scene.title" in lesson[1]
    assert "item.dataset.status = scene.status" in lesson[1]
    assert "cues.position().scene_id" in lesson[1]
    assert '"Preparing a lesson on " + card.subject' in client
    sheet = re.search(r"<style>(.*?)</style>", index, re.DOTALL)
    assert sheet is not None
    reduced = re.search(
        r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n  \}", sheet[1], re.DOTALL
    )
    assert reduced is not None
    assert '.lesson li[data-status="building"]::before { animation: none; }' in reduced[1]


def test_the_check_page_probes_the_lesson_payloads() -> None:
    text = VISUAL_CHECK.read_text()
    good = re.search(r"const good = \[(.*?)\n\];", text, re.DOTALL)
    hostile = re.search(r"const hostile = \[(.*?)\n\];", text, re.DOTALL)
    assert good is not None and hostile is not None
    for name in (
        '"lesson.attach"',
        '"lesson.cue step"',
        '"lesson.cue scene"',
        '"lesson.sync"',
        '"lesson.state"',
    ):
        assert name in good[1], name
    for name in (
        '"lesson.cue extra key"',
        '"lesson.cue tag kind set"',
        '"lesson.cue step n 6"',
        '"lesson.cue scene n 13"',
        '"lesson.cue scene_id Bad"',
        '"lesson.cue barrier -1"',
        '"lesson.cue epoch 0"',
        '"lesson.cue chunk_id -1"',
        '"lesson.cue audio_ms 1.5"',
        '"lesson.sync barrier 0"',
        '"lesson.state thirteen scenes"',
        '"lesson.state status drawn"',
        '"lesson.state current Bad"',
    ):
        assert name in hostile[1], name


def test_a_caption_for_a_turn_with_no_learner_line_keeps_its_record() -> None:
    client = CLIENT.read_text()
    caption = re.search(r'onPayload\("caption", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL)
    assert caption is not None
    body = caption[1]
    assert 'turn = { turn_id: payload.turn_id, learner: "", tutor: [] };' in body
    assert body.index("if (turn === undefined) {") < body.index("turn.tutor.push(payload.text);")

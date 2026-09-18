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
GUARDED = [CLIENT, VISUALS, FRAME, LESSON]
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
    for path in (CLIENT, VISUALS, LESSON):
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
    for path in (INDEX, VISUAL_CHECK):
        assert host_policy(path.read_text()) == HOST_POLICY, path.name


def test_the_host_policy_precedes_every_script_and_style() -> None:
    for path in (INDEX, VISUAL_CHECK):
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
    assert '"state": ["type", "seq", "state", "phase", "interrupted"]' in keys[1]
    assert '"caption": ["type", "seq", "turn_id", "text", "lead_ms"]' in keys[1]
    assert '"visual.pending": ["type", "seq", "turn_id", "title"]' in keys[1]


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
        '"visual.pending title of 81"',
        '"visual.pending missing title"',
        '"visual.pending turn_id of 33"',
    ):
        assert name in hostile[1], name


def test_a_pending_visual_reaches_the_pending_listener() -> None:
    text = VISUALS.read_text()
    receive = re.search(r"export function receive\(payload\) \{(.*?)\n\}", text, re.DOTALL)
    validate = re.search(r"export function validate\(payload\) \{(.*?)\n\}", text, re.DOTALL)
    assert receive is not None and validate is not None
    assert re.search(
        r'case "visual\.pending":\s*listeners\.get\("pending"\)\?\.\(payload\);', receive[1]
    )
    case = re.search(r'case "visual\.pending":(.*?)return true;', validate[1], re.DOTALL)
    assert case is not None
    assert 'cappedString(payload, "turn_id")' in case[1]
    assert 'cappedString(payload, "title")' in case[1]


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
    assert ".canvas iframe.landing" in index and ".canvas.drawing" in index


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
    assert ".canvas.drawing { animation: none;" in reduced[1]
    assert sheet[1][reduced.end() :].strip() == ""
    for selector in (".canvas iframe {", ".canvas iframe.landing {", ".canvas.drawing {"):
        assert sheet[1].index(selector) < reduced.start(), selector


def test_the_drawing_state_clears_on_a_landing_and_a_matching_clear_only() -> None:
    client = CLIENT.read_text()
    pending = re.search(r'onPayload\("pending", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL)
    history = re.search(
        r'onPayload\("history", \(items, current\) => \{(.*?)\n\}\);', client, re.DOTALL
    )
    assert pending is not None and history is not None
    assert "payload.turn_id !== pendingTurn" in pending[1]
    assert "items.length > historyCount || current < 0" in history[1]
    assert 'canvas.classList.remove("drawing")' in pending[1]
    assert 'canvas.classList.remove("drawing")' in history[1]
    assert 'onJson("visual.pending", receive)' in client


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
    assert len(receives) == 5
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


def test_only_the_thinking_state_sets_a_turns_phase() -> None:
    client = CLIENT.read_text()
    state = re.search(r'onPayload\("state", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL)
    transcript = re.search(
        r'onPayload\("transcript", \(payload\) => \{(.*?)\n\}\);', client, re.DOTALL
    )
    assert state is not None and transcript is not None
    writes = [line for line in state[1].splitlines() if "turns" in line]
    assert len(writes) == 1
    assert writes[0].index('payload.state === "thinking"') < writes[0].index(".phase =")
    assert '"speaking"' not in writes[0]
    assert '"listening"' not in writes[0]
    opened = re.search(r"turns\.push\(\{(.*?)\}\)", transcript[1], re.DOTALL)
    assert opened is not None
    assert "phase: null" in opened[1]
    assert "turn_id: payload.turn_id" in opened[1]
    assert "tutor: []" in opened[1]


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

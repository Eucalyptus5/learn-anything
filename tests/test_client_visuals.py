import re

from tutor.signaling import CLIENT_ROOT

CLIENT = CLIENT_ROOT / "client.js"
VISUALS = CLIENT_ROOT / "visuals.js"
FRAME = CLIENT_ROOT / "frame.html"
INDEX = CLIENT_ROOT / "index.html"
VISUAL_CHECK = CLIENT_ROOT / "visual_check.html"
GUARDED = [CLIENT, VISUALS, FRAME]
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
    for path in (CLIENT, VISUALS):
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
    for path in (CLIENT, VISUALS):
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

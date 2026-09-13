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


def host_policy(text: str) -> dict[str, list[str]]:
    metas = POLICY_META.findall(text)
    assert len(metas) == 1
    directives = [directive.split() for directive in metas[0].split(";") if directive.strip()]
    assert len(directives) == len(HOST_POLICY)
    return {name: sources for name, *sources in directives}


def test_sandbox_attribute_is_exactly_allow_scripts() -> None:
    texts = {path.name: path.read_text() for path in GUARDED}
    sandboxes = re.findall(r'setAttribute\("sandbox",\s*"([^"]*)"\)', texts["visuals.js"])
    assert sandboxes == ["allow-scripts", "allow-scripts"]
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

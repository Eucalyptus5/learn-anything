import asyncio
import logging
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from tutor.tools.models import PATH_EXTENSIONS
from tutor.tools.ripgrep import reap

logger = logging.getLogger(__name__)

MAX_GLOBS = 6
WALK_TIMEOUT_MS = 2000
READ_CHUNK_BYTES = 65536

TOKEN_TRIM = "\"'`()[]{}<>,.;:!?"
WORD = re.compile(r"[a-z]+")

# "go" and "c" are ordinary English words in a spoken turn; the tree walk covers those trees.
LANGUAGE_SUFFIXES: dict[str, list[str]] = {
    "golang": [".go"],
    "python": [".py", ".pyi"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "rust": [".rs"],
    "java": [".java"],
    "kotlin": [".kt"],
    "ruby": [".rb"],
    "swift": [".swift"],
    "html": [".html"],
    "css": [".css"],
    "sql": [".sql"],
    "markdown": [".md"],
    "yaml": [".yaml", ".yml"],
    "toml": [".toml"],
    "json": [".json"],
}

_walk_cache: dict[Path, list[str]] = {}


def _dedup(globs: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for glob in globs:
        seen[glob] = None
    return list(seen)


def _named_files(user_text: str) -> list[str]:
    globs = []
    for token in user_text.split():
        path = PurePosixPath(token.strip(TOKEN_TRIM))
        suffix = path.suffix.lower()
        if suffix not in PATH_EXTENSIONS or not path.stem:
            continue
        parent = path.parent
        globs.append(f"**/*{suffix}" if str(parent) == "." else f"{parent}/**/*{suffix}")
    return _dedup(globs)


def _language_words(user_text: str) -> list[str]:
    globs = []
    for word in WORD.findall(user_text.lower()):
        for suffix in LANGUAGE_SUFFIXES.get(word, ()):
            globs.append(f"**/*{suffix}")
    return _dedup(globs)


def _ranked(counts: Counter[str]) -> list[str]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"**/*{suffix}" for suffix, _ in ordered]


async def _walk(root: Path) -> list[str]:
    if root in _walk_cache:
        return _walk_cache[root]

    proc = await asyncio.create_subprocess_exec(
        "rg",
        "--files",
        "--no-require-git",
        cwd=root,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    counts: Counter[str] = Counter()
    buffer = b""
    try:
        async with asyncio.timeout(WALK_TIMEOUT_MS / 1000):
            while True:
                chunk = await proc.stdout.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                buffer += chunk
                while (newline := buffer.find(b"\n")) >= 0:
                    raw, buffer = buffer[:newline], buffer[newline + 1 :]
                    suffix = PurePosixPath(raw.decode(errors="replace")).suffix.lower()
                    if suffix in PATH_EXTENSIONS:
                        counts[suffix] += 1
    except TimeoutError:
        await reap(proc)
        logger.warning("walk_timeout root=%s suffixes=%d", root, len(counts))
        return _ranked(counts)
    except asyncio.CancelledError:
        await reap(proc)
        raise

    await proc.wait()
    _walk_cache[root] = _ranked(counts)
    return _walk_cache[root]


async def derive_globs(user_text: str, root: Path) -> list[str]:
    globs = _named_files(user_text) or _language_words(user_text) or await _walk(root)
    return globs[:MAX_GLOBS]

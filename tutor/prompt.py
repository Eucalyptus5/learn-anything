import asyncio
import logging
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from tutor.tools.models import PATH_EXTENSIONS
from tutor.tools.ripgrep import READ_CHUNK_BYTES, reap

logger = logging.getLogger(__name__)

MAX_GLOBS = 6
WALK_TIMEOUT_MS = 2000

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


def _named_files(user_text: str) -> list[str]:
    globs = []
    for token in user_text.split():
        path = PurePosixPath(token.strip(TOKEN_TRIM))
        if path.suffix not in PATH_EXTENSIONS or not path.stem:
            continue
        parent = path.parent
        globs.append(f"**/*{path.suffix}" if str(parent) == "." else f"{parent}/**/*{path.suffix}")
    return list(dict.fromkeys(globs))


def _language_words(user_text: str) -> list[str]:
    globs = []
    for word in WORD.findall(user_text.lower()):
        for suffix in LANGUAGE_SUFFIXES.get(word, ()):
            globs.append(f"**/*{suffix}")
    return list(dict.fromkeys(globs))


def _ranked(counts: Counter[str]) -> list[str]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"**/*{suffix}" for suffix, _ in ordered] or ["**/*"]


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
        try:
            async with asyncio.timeout(WALK_TIMEOUT_MS / 1000):
                while True:
                    chunk = await proc.stdout.read(READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    *lines, buffer = (buffer + chunk).split(b"\n")
                    for line in lines:
                        suffix = PurePosixPath(line.decode(errors="replace")).suffix
                        if suffix in PATH_EXTENSIONS:
                            counts[suffix] += 1
        except TimeoutError:
            await reap(proc)
            logger.warning("walk_timeout root=%s suffixes=%d", root, len(counts))
            return _ranked(counts)
        await proc.wait()
    except asyncio.CancelledError:
        await reap(proc)
        raise

    _walk_cache[root] = _ranked(counts)
    return _walk_cache[root]


async def derive_globs(user_text: str, root: Path) -> list[str]:
    globs = _named_files(user_text) or _language_words(user_text) or await _walk(root)
    return globs[:MAX_GLOBS]

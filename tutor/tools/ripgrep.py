import asyncio
import base64
import json
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

from tutor.tools.models import MAX_COLUMNS, SearchBudget

logger = logging.getLogger(__name__)

READ_CHUNK_BYTES = 65536
VISIBILITY_WALK_TIMEOUT_MS = 5000


class RipgrepUnavailable(ValueError):
    pass


class RipgrepFailed(RuntimeError):
    pass


def probe_ripgrep(minimum: tuple[int, int, int] = (14, 0, 0)) -> str:
    try:
        result = subprocess.run(["rg", "--version"], capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RipgrepUnavailable("rg binary not found on PATH") from exc

    version = result.stdout.splitlines()[0].split()[1]
    found = tuple(int(part) for part in version.split(".")[:3])
    if found < minimum:
        minimum_str = ".".join(str(part) for part in minimum)
        raise RipgrepUnavailable(
            f"rg version {version} is below the required minimum {minimum_str}"
        )
    return version


def decode_field(field: dict[str, str]) -> str | None:
    if "text" in field:
        return field["text"]
    try:
        return base64.b64decode(field["bytes"]).decode()
    except UnicodeDecodeError:
        return None


def visible_files(root: Path) -> frozenset[str]:
    try:
        walk = subprocess.run(
            ["rg", "--files", "--no-require-git", "--null", "."],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=VISIBILITY_WALK_TIMEOUT_MS / 1000,
        )
    except subprocess.TimeoutExpired as exc:
        raise RipgrepFailed(f"rg --files timed out after {VISIBILITY_WALK_TIMEOUT_MS}ms") from exc

    # NUL is the one byte a path cannot hold, so a filename carrying a line or record separator
    # cannot split itself into a second entry.
    entries = walk.stdout.split(b"\x00")
    if entries[-1]:
        # A walk cut off mid-write leaves a prefix of a path, and that prefix names another file.
        entries.pop()

    names: set[str] = set()
    for raw in entries:
        if not raw:
            continue
        try:
            names.add(raw.decode())
        except UnicodeDecodeError:
            continue

    # rg --files exits 1 when the tree holds nothing visible, which is not a failure.
    if walk.returncode not in (0, 1):
        detail = walk.stderr.decode(errors="replace").strip()
        if not names:
            raise RipgrepFailed(f"rg --files exited {walk.returncode}: {detail}")
        logger.warning("ripgrep_walk_partial exit=%d stderr=%s", walk.returncode, detail)

    return frozenset(names)


def record_size(record: dict) -> int:
    return len(json.dumps(record, separators=(",", ":")).encode())


def _clip(record: dict) -> None:
    data = record["data"]
    if "text" not in data["lines"]:
        return

    data["lines"]["text"] = data["lines"]["text"][:MAX_COLUMNS]
    limit = len(data["lines"]["text"].encode())
    kept = []
    for submatch in data["submatches"]:
        if submatch["start"] >= limit:
            continue
        submatch["end"] = min(submatch["end"], limit)
        if "text" in submatch["match"]:
            span = submatch["end"] - submatch["start"]
            submatch["match"]["text"] = submatch["match"]["text"].encode()[:span].decode()
        kept.append(submatch)
    data["submatches"] = kept


async def reap(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        proc.kill()
    # A reader left over its limit has paused its transport, which then never reports EOF, and
    # wait() blocks until every pipe has reported it.
    for pipe in (proc.stdout, proc.stderr):
        if pipe is None:
            continue
        while await pipe.read(READ_CHUNK_BYTES):
            pass
    await proc.wait()


async def run_ripgrep(
    query: str, globs: Sequence[str], root: Path, budget: SearchBudget
) -> tuple[list[dict], int, bool, bool]:
    if not globs:
        raise ValueError("run_ripgrep needs at least one glob")

    visible = await asyncio.to_thread(visible_files, root)

    argv = [
        "rg",
        "--json",
        "--context",
        str(budget.context_lines),
        "--no-require-git",
    ]
    for glob in globs:
        argv += ["--glob", glob]
    argv += ["-e", query, "."]

    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=root,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    records: list[dict] = []
    byte_count = 0
    truncated = False
    oversized = False
    buffer = b""
    skipping = False
    block: str | None = None

    try:
        try:
            async with asyncio.timeout(budget.timeout_ms / 1000):
                while True:
                    chunk = await proc.stdout.read(READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    buffer += chunk

                    while True:
                        newline = buffer.find(b"\n")
                        if newline < 0:
                            if skipping:
                                buffer = b""
                            break
                        raw, buffer = buffer[:newline], buffer[newline + 1 :]
                        if skipping:
                            skipping = False
                            continue
                        if len(raw) > budget.max_record_bytes:
                            # An oversized record is never parsed, so the begin record that
                            # opened its block is the only thing naming the file it came from.
                            oversized = oversized or block in visible
                            continue
                        record = json.loads(raw)
                        if record["type"] == "begin":
                            block = decode_field(record["data"]["path"])
                            continue
                        if record["type"] not in ("match", "context"):
                            continue
                        if decode_field(record["data"]["path"]) not in visible:
                            continue
                        _clip(record)
                        size = record_size(record)
                        if byte_count + size > budget.max_bytes:
                            truncated = True
                            break
                        records.append(record)
                        byte_count += size

                    if truncated:
                        break
                    if len(buffer) > budget.max_record_bytes:
                        oversized = oversized or block in visible
                        skipping = True
                        buffer = b""
        except TimeoutError:
            truncated = True

        if truncated:
            await reap(proc)
            return records, byte_count, True, oversized

        stderr = await proc.stderr.read()
        code = await proc.wait()
    except asyncio.CancelledError:
        await reap(proc)
        raise

    if code >= 2:
        detail = stderr.decode(errors="replace").strip()
        if not any(record["type"] == "match" for record in records):
            raise RipgrepFailed(f"rg exited {code}: {detail}")
        logger.warning("ripgrep_partial exit=%d stderr=%s", code, detail)

    return records, byte_count, False, oversized

import asyncio
import json
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

from tutor.tools.models import MAX_COLUMNS, SearchBudget

logger = logging.getLogger(__name__)

READ_CHUNK_BYTES = 65536


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


def _clip(record: dict) -> None:
    data = record["data"]
    if "text" not in data["lines"]:
        return

    data["lines"]["text"] = data["lines"]["text"][:MAX_COLUMNS]
    kept = []
    for submatch in data["submatches"]:
        if submatch["start"] >= MAX_COLUMNS:
            continue
        submatch["end"] = min(submatch["end"], MAX_COLUMNS)
        span = submatch["end"] - submatch["start"]
        submatch["match"]["text"] = submatch["match"]["text"][:span]
        kept.append(submatch)
    data["submatches"] = kept


async def _reap(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        proc.kill()
    await proc.wait()


async def run_ripgrep(
    query: str, globs: Sequence[str], root: Path, budget: SearchBudget
) -> tuple[list[dict], int, bool, bool]:
    if not globs:
        raise ValueError("run_ripgrep needs at least one glob")

    argv = [
        "rg",
        "--json",
        "--sort",
        "path",
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
                            oversized = True
                            continue
                        record = json.loads(raw)
                        if record["type"] not in ("match", "context"):
                            continue
                        _clip(record)
                        size = len(json.dumps(record, separators=(",", ":")).encode())
                        if byte_count + size > budget.max_bytes:
                            truncated = True
                            break
                        records.append(record)
                        byte_count += size

                    if truncated:
                        break
                    if len(buffer) > budget.max_record_bytes:
                        oversized = True
                        skipping = True
                        buffer = b""
        except TimeoutError:
            truncated = True

        if truncated:
            await _reap(proc)
            return records, byte_count, True, oversized

        stderr = await proc.stderr.read()
        code = await proc.wait()
    except asyncio.CancelledError:
        await _reap(proc)
        raise

    if code >= 2:
        detail = stderr.decode(errors="replace").strip()
        if not any(record["type"] == "match" for record in records):
            raise RipgrepFailed(f"rg exited {code}: {detail}")
        logger.warning("ripgrep_partial exit=%d stderr=%s", code, detail)

    return records, byte_count, False, oversized

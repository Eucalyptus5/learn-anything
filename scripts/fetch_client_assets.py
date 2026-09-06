"""Fetches the pinned mermaid UMD bundle into client/vendor/mermaid.min.js and checks it
against a pinned sha256.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENDOR = REPO / "client" / "vendor"

MERMAID_VERSION = "12.0.0"
MERMAID_URL = f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/mermaid.min.js"
MERMAID_SHA256 = "28fca7ae6ebc7ed7bb63bde63136a74bfef14f296a57e403657eeb8b32836073"
MERMAID_PATH = VENDOR / "mermaid.min.js"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    MERMAID_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MERMAID_PATH.exists() and sha256_of(MERMAID_PATH) == MERMAID_SHA256:
        size = MERMAID_PATH.stat().st_size
        print(f"present  {MERMAID_PATH.resolve()}  {size} bytes")
        return

    tmp_path = MERMAID_PATH.with_suffix(MERMAID_PATH.suffix + ".part")
    with urllib.request.urlopen(MERMAID_URL) as response, tmp_path.open("wb") as f:
        f.write(response.read())

    digest = sha256_of(tmp_path)
    if digest != MERMAID_SHA256:
        tmp_path.unlink()
        raise SystemExit(f"sha256 mismatch: expected {MERMAID_SHA256}, got {digest}")

    os.replace(tmp_path, MERMAID_PATH)
    size = MERMAID_PATH.stat().st_size
    print(f"downloaded  {MERMAID_PATH.resolve()}  {size} bytes")


if __name__ == "__main__":
    main()

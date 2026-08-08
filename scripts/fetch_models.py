"""Fetches the Silero VAD ONNX weights into models/silero/silero_vad.onnx and checks them
against a pinned sha256. Whisper weights are not fetched here: tutor.stt.load_whisper passes its
model_dir as download_root and faster-whisper pulls Systran/faster-whisper-base.en into it on
first construction; the callers point it at models/whisper.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"

SILERO_TAG = "v6.2.1"
SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    f"{SILERO_TAG}/src/silero_vad/data/silero_vad.onnx"
)
SILERO_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_PATH = MODELS / "silero" / "silero_vad.onnx"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    SILERO_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SILERO_PATH.exists() and sha256_of(SILERO_PATH) == SILERO_SHA256:
        size = SILERO_PATH.stat().st_size
        print(f"present  {SILERO_PATH.resolve()}  {size} bytes")
        return

    tmp_path = SILERO_PATH.with_suffix(SILERO_PATH.suffix + ".part")
    with urllib.request.urlopen(SILERO_URL) as response, tmp_path.open("wb") as f:
        f.write(response.read())

    digest = sha256_of(tmp_path)
    if digest != SILERO_SHA256:
        tmp_path.unlink()
        raise SystemExit(f"sha256 mismatch: expected {SILERO_SHA256}, got {digest}")

    os.replace(tmp_path, SILERO_PATH)
    size = SILERO_PATH.stat().st_size
    print(f"downloaded  {SILERO_PATH.resolve()}  {size} bytes")


if __name__ == "__main__":
    main()

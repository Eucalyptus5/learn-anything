"""Fetches the pinned browser libraries the visual frame loads into client/vendor and checks
each download against a pinned sha256 before anything lands in place.
"""

import hashlib
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
VENDOR = REPO / "client" / "vendor"
NPM = "https://registry.npmjs.org"

KATEX_FAMILIES = (
    "KaTeX_AMS-Regular",
    "KaTeX_Caligraphic-Bold",
    "KaTeX_Caligraphic-Regular",
    "KaTeX_Fraktur-Bold",
    "KaTeX_Fraktur-Regular",
    "KaTeX_Main-Bold",
    "KaTeX_Main-BoldItalic",
    "KaTeX_Main-Italic",
    "KaTeX_Main-Regular",
    "KaTeX_Math-BoldItalic",
    "KaTeX_Math-Italic",
    "KaTeX_SansSerif-Bold",
    "KaTeX_SansSerif-Italic",
    "KaTeX_SansSerif-Regular",
    "KaTeX_Script-Regular",
    "KaTeX_Size1-Regular",
    "KaTeX_Size2-Regular",
    "KaTeX_Size3-Regular",
    "KaTeX_Size4-Regular",
    "KaTeX_Typewriter-Regular",
)
KATEX_FONTS = {
    f"package/dist/fonts/{family}.{suffix}": VENDOR / "katex" / "fonts" / f"{family}.{suffix}"
    for family in KATEX_FAMILIES
    for suffix in ("ttf", "woff", "woff2")
}


class Asset(NamedTuple):
    name: str
    url: str
    sha256: str
    members: dict[str, Path]  # tar member -> destination; {"": path} for a bare file


ASSETS = [
    Asset(
        name="mermaid",
        url="https://cdn.jsdelivr.net/npm/mermaid@12.0.0/dist/mermaid.min.js",
        sha256="28fca7ae6ebc7ed7bb63bde63136a74bfef14f296a57e403657eeb8b32836073",
        members={"": VENDOR / "mermaid.min.js"},
    ),
    Asset(
        name="plotly",
        url=f"{NPM}/plotly.js-dist-min/-/plotly.js-dist-min-4.1.0.tgz",
        sha256="536085ec2cbfdaa8d9b2ca1f41d71539daee2f956a49d5939f381938dd27e075",
        members={"package/plotly.min.js": VENDOR / "plotly.min.js"},
    ),
    Asset(
        name="katex",
        url=f"{NPM}/katex/-/katex-0.18.7.tgz",
        sha256="9a80a3fba2367e99bf67b52bfff52e9534c8c7f198e4eaa12699e4f1df9a0bda",
        members={
            "package/dist/katex.min.js": VENDOR / "katex" / "katex.min.js",
            "package/dist/katex.min.css": VENDOR / "katex" / "katex.min.css",
            **KATEX_FONTS,
        },
    ),
    Asset(
        name="p5",
        url=f"{NPM}/p5/-/p5-2.3.3.tgz",
        sha256="c13125922d6ffca04d8b5c8623b98974babeacabe794ac21c5428c4b3fa5632d",
        members={"package/lib/p5.min.js": VENDOR / "p5.min.js"},
    ),
    Asset(
        name="d3",
        url=f"{NPM}/d3/-/d3-7.9.0.tgz",
        sha256="7e36605710a2ba54846797c8c6d888911341b215ae53257dc32a49a0b824355e",
        members={"package/dist/d3.min.js": VENDOR / "d3.min.js"},
    ),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, members: dict[str, Path]) -> None:
    with tarfile.open(archive) as tar:
        for member, destination in members.items():
            info = tar.getmember(member)
            if not info.isfile():
                raise SystemExit(f"{member} is not a regular file")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(info) as source, destination.open("wb") as target:
                target.write(source.read())


def fetch(asset: Asset) -> None:
    first = next(iter(asset.members.values()))
    first.parent.mkdir(parents=True, exist_ok=True)
    part = first.with_name(first.name + ".part")
    with urllib.request.urlopen(asset.url) as response, part.open("wb") as target:
        shutil.copyfileobj(response, target)
    digest = sha256_of(part)
    if digest != asset.sha256:
        part.unlink()
        raise SystemExit(f"{asset.name}: sha256 mismatch, expected {asset.sha256}, got {digest}")
    if "" in asset.members:
        os.replace(part, first)
        return
    extract(part, asset.members)
    part.unlink()
    (VENDOR / f"{asset.name}.sha256").write_text(asset.sha256)


def main() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        if "" in asset.members:
            verified = sha256_of(asset.members[""]) if asset.members[""].is_file() else ""
        else:
            stamp = VENDOR / f"{asset.name}.sha256"
            verified = stamp.read_text() if stamp.is_file() else ""
        present = verified == asset.sha256 and all(
            destination.is_file() for destination in asset.members.values()
        )
        if present:
            status = "present"
        else:
            fetch(asset)
            status = "downloaded"
        size = sum(destination.stat().st_size for destination in asset.members.values())
        print(f"{status:<10}  {asset.name}  {size} bytes")


if __name__ == "__main__":
    main()

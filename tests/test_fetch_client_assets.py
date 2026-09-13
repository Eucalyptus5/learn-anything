import importlib.util
import io
import re
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_client_assets.py"
_spec = importlib.util.spec_from_file_location("fetch_client_assets", SCRIPT)
fetch_client_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_client_assets)

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def write_tarball(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_every_asset_has_a_sixty_four_hex_sha256() -> None:
    for asset in fetch_client_assets.ASSETS:
        assert SHA256.match(asset.sha256), asset.name


def test_members_land_under_vendor_and_never_above_it() -> None:
    vendor = fetch_client_assets.VENDOR.resolve()
    for asset in fetch_client_assets.ASSETS:
        assert asset.members, asset.name
        for destination in asset.members.values():
            assert destination.resolve().is_relative_to(vendor), destination


def test_extract_writes_only_the_listed_members(tmp_path: Path) -> None:
    tar_path = tmp_path / "pkg.tgz"
    write_tarball(
        tar_path,
        {"package/a.js": b"a;", "package/b.js": b"b;", "package/../evil.js": b"evil;"},
    )

    fetch_client_assets.extract(tar_path, {"package/a.js": tmp_path / "out" / "a.js"})

    assert (tmp_path / "out" / "a.js").read_bytes() == b"a;"
    written = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert written == {"pkg.tgz", "a.js"}


def test_a_sha256_mismatch_deletes_the_download_and_raises(tmp_path: Path) -> None:
    source = tmp_path / "source.js"
    source.write_bytes(b"not the pinned bytes")
    destination = tmp_path / "out" / "lib.js"
    asset = fetch_client_assets.Asset(
        name="lib",
        url=source.as_uri(),
        sha256="0" * 64,
        members={"": destination},
    )

    with pytest.raises(SystemExit):
        fetch_client_assets.fetch(asset)

    assert not destination.exists()
    assert list(tmp_path.rglob("*.part")) == []
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [source]


def test_a_changed_pin_refetches_an_extracted_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tar_path = tmp_path / "lib.tgz"
    write_tarball(tar_path, {"package/x.js": b"x;"})
    vendor = tmp_path / "vendor"
    asset = fetch_client_assets.Asset(
        name="lib",
        url=tar_path.as_uri(),
        sha256=fetch_client_assets.sha256_of(tar_path),
        members={"package/x.js": vendor / "x.js"},
    )
    monkeypatch.setattr(fetch_client_assets, "VENDOR", vendor)
    monkeypatch.setattr(fetch_client_assets, "ASSETS", [asset])

    fetch_client_assets.main()
    assert capsys.readouterr().out.startswith("downloaded")
    fetch_client_assets.main()
    assert capsys.readouterr().out.startswith("present")

    monkeypatch.setattr(fetch_client_assets, "ASSETS", [asset._replace(sha256="0" * 64)])
    with pytest.raises(SystemExit):
        fetch_client_assets.main()
    assert (vendor / "x.js").read_bytes() == b"x;"

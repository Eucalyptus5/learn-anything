import subprocess


class RipgrepUnavailable(ValueError):
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

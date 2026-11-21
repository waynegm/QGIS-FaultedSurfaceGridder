"""Build the QGIS plugin distribution zip."""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = ROOT / "faulted_surface_gridder"
DIST_DIR = ROOT / "dist"

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pyd"}


def plugin_version() -> str:
    metadata = (PLUGIN_DIR / "metadata.txt").read_text(encoding="utf-8")
    match = re.search(r"^version\s*=\s*(\S+)", metadata, re.MULTILINE)
    if match is None:
        raise ValueError("faulted_surface_gridder/metadata.txt has no version entry")
    return match.group(1)


def build() -> Path:
    DIST_DIR.mkdir(exist_ok=True)
    output = DIST_DIR / f"faulted_surface_gridder-{plugin_version()}.zip"

    directories = {"faulted_surface_gridder/"}
    files: list[tuple[Path, str]] = []
    for path in sorted(PLUGIN_DIR.rglob("*")):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.is_dir():
            directories.add(f"{relative}/")
        elif path.is_file() and path.suffix not in EXCLUDED_SUFFIXES:
            files.append((path, relative))

    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for directory in sorted(directories):
            archive.writestr(directory, b"")
        for path, relative in files:
            archive.write(path, relative)

    return output


if __name__ == "__main__":
    print(f"Wrote {build().relative_to(ROOT)}")

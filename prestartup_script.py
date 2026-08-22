"""ComfyUI-UniRig Prestartup Script."""

import logging
import shutil
from pathlib import Path

import folder_paths
from comfy_env import setup_env

log = logging.getLogger("unirig")

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS = SCRIPT_DIR / "assets"

# The CONFIGURED input directory, never the code-tree one. ComfyUI Desktop
# (--base-directory) and --input-directory both relocate it, and the load
# nodes only ever scan folder_paths.get_input_directory() -- so seeding into
# <comfyui>/input there puts files somewhere nothing reads. main.py runs
# apply_custom_paths() before prestartup scripts, so this is already resolved.
# No try/except fallback: if folder_paths is missing we are not inside ComfyUI
# and there is nothing sensible to seed. execute_prestartup_script() catches
# and logs, so a failure here fails this pack loudly without breaking startup.
INPUT = Path(folder_paths.get_input_directory())


def copy_files(src: Path, dst: Path, pattern: str = "*") -> int:
    """Copy bundled assets into a directory. Returns files written.

    Seeds rather than syncs: an existing file is left alone, so a user's
    edited demo asset survives every relaunch. Raises if `src` is missing --
    a typo'd asset directory is a packaging bug, and silence is how it stays
    one.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"asset directory not found: {src}")
    written = 0
    for f in src.glob(pattern):
        if not f.is_file():
            continue
        target = dst / f.relative_to(src)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        written += 1
    return written


# Viewer JS is vendored in javascript/ and served by ComfyUI via
# [tool.comfy] web -- nothing to copy at startup.
_seeded = copy_files(ASSETS, INPUT / "3d")
_seeded += copy_files(ASSETS / "animation_templates", INPUT / "animation_templates", "**/*")
_seeded += copy_files(ASSETS / "animation_characters", INPUT / "animation_characters")
if _seeded:
    log.info("seeded %d asset(s) into %s", _seeded, INPUT)

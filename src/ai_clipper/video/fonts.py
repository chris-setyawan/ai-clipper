"""
Font resolution for burned-in captions.

A handful of open-licence fonts ship inside the project so that a clip renders
identically on any machine, with no download step and nothing to install. The
renderer points libass at that directory first; anything installed on the system
still works too, so a user with their own brand font is not locked out.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

def _assets_dir() -> Path:
    """
    Where the bundled fonts live.

    Inside the package, so a wheel carries them: an installed copy has no repo
    root to look up from. The old location beside the source tree is still
    checked, because a checkout from before the move should not silently lose
    its fonts and fall back to whatever face libass finds.
    """
    inside = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    if inside.is_dir():
        return inside
    return Path(__file__).resolve().parents[3] / "assets" / "fonts"


ASSETS_DIR = _assets_dir()

# Family name as libass sees it -> the file it comes from. The family name is
# what goes in the ASS style line, and it must match the font's internal name,
# not the filename.
BUNDLED = {
    "Anton": "Anton-Regular.ttf",
    "Montserrat ExtraBold": "Montserrat-ExtraBold.ttf",
    "Bebas Neue": "BebasNeue-Regular.ttf",
}

DEFAULT_FONT = "Montserrat ExtraBold"


def fonts_dir() -> str:
    """Directory handed to libass via ffmpeg's fontsdir option."""
    return str(ASSETS_DIR)


def bundled_fonts() -> List[str]:
    """Families that ship with the project and are guaranteed to render."""
    return [name for name, f in BUNDLED.items() if (ASSETS_DIR / f).exists()]


def missing_bundled() -> List[str]:
    return [name for name, f in BUNDLED.items() if not (ASSETS_DIR / f).exists()]


def resolve(family: str) -> str:
    """
    Return the family name to write into the ASS file.

    A bundled family passes through. Anything else is assumed to be installed on
    the system; libass falls back to a default face if it isn't, which is why
    check_font below exists for callers that want to warn first.
    """
    return family or DEFAULT_FONT


def is_bundled(family: str) -> bool:
    return family in BUNDLED and (ASSETS_DIR / BUNDLED[family]).exists()


def check_font(family: str) -> tuple:
    """
    (ok, message). Bundled fonts are always ok. System fonts are reported as
    unverified rather than missing, because we cannot enumerate them portably
    without pulling in another dependency.
    """
    if is_bundled(family):
        return True, f"'{family}' ships with the project"
    return False, (
        f"'{family}' is not bundled - it must be installed on this machine, "
        f"or captions will fall back to a default face. "
        f"Bundled options: {', '.join(bundled_fonts())}"
    )

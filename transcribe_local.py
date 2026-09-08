"""
Transcribe from a clone, without installing anything.

    python transcribe_local.py "podcast.mov" --model large-v3 --device cuda

The real implementation lives in the package (ai_clipper.cli.transcribe), where
an installed copy calls it `ai-clipper-transcribe`. This file exists because the
script is often run from wherever the video is rather than from inside the
project, so it goes looking for src/ first.
"""

import os
import sys


# the file that proves a src/ directory holds a current copy of the package
_MARKER = os.path.join("ai_clipper", "asr", "corrections.py")


def _is_usable(src_dir) -> bool:
    """
    A directory only counts if it contains the module we actually need.

    Checking merely for src/ai_clipper is not enough: an older extracted copy
    of the project has that directory but not the corrections module, and
    finding it first produced a ModuleNotFoundError several folders away from
    anything the user could see was wrong.
    """
    return os.path.isfile(os.path.join(src_dir, _MARKER))


def _find_src():
    """
    Locate a usable src/ directory.

    The script is meant to run from anywhere - usually next to the video rather
    than inside the project - so it searches upward from itself and from the
    working directory, and one level into each sibling folder, skipping any copy
    that is missing the module.
    """
    for base in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        current = base
        for _ in range(4):
            found = _search_below(current, max_depth=3)
            if found:
                return found
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    return None


def _search_below(root, max_depth=3):
    """
    Look for a usable src/ within max_depth levels of root, newest first.

    Nested extracted copies are the normal state of a project folder that has
    been unzipped a few times, so guessing at fixed layouts does not hold up -
    this walks, skips the obvious noise, and prefers the most recently modified
    match so a fresh extraction wins over an old one.
    """
    if not os.path.isdir(root):
        return None

    skip = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}
    matches = []
    root_depth = root.rstrip(os.sep).count(os.sep)

    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        if dirpath.count(os.sep) - root_depth >= max_depth:
            dirnames[:] = []
            continue
        candidate = os.path.join(dirpath, "src")
        if _is_usable(candidate):
            matches.append(candidate)

    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getmtime(os.path.join(p, _MARKER)))




def main():
    src = _find_src()
    if src is None:
        print("ERROR: could not find the project's src/ directory from here.")
        print("       Put this script inside the ai-clipper folder, or install")
        print("       the project (pip install .) and use ai-clipper-transcribe.")
        sys.exit(1)
    sys.path.insert(0, src)
    from ai_clipper.cli.transcribe import main as run

    run()


if __name__ == "__main__":
    main()

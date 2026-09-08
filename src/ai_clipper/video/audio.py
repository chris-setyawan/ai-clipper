"""
Make every clip land at the loudness the platforms expect.

A podcast is mastered for headphones and a long sitting. Short-form is not:
TikTok, Reels and Shorts all normalise what you upload towards roughly -14 LUFS,
and a clip that arrives quieter is turned up by the platform along with its room
tone, while one that arrives hot is turned down and sounds flat next to whatever
plays after it. Cutting a clip out of a podcast without touching the audio means
shipping whatever level that particular minute happened to sit at.

Two decisions here are worth stating, because the obvious implementation gets
both of them wrong.

**Measure the episode, not the clip.** Running ffmpeg's `loudnorm` per clip is
the usual answer and it is the wrong one. It normalises each clip to the target
independently, so a whisper and a shout come out at the same level and the show
loses the dynamics that made the moment worth clipping. Measuring the source
once and applying the same offset to all of its clips keeps the relationship
between them, and costs one pass over the audio instead of one per output file.

**Gain alone cannot do it.** Conversational audio that has never been compressed
often sits at -20 LUFS with peaks already close to full scale: there is no
headroom to raise it into. Clamping the gain to whatever the peak allows, which
is what a careful implementation does first, produces no change at all on
exactly the files that needed it most. So the gain goes in ahead of a limiter,
which catches the handful of peaks that would clip, and the ceiling is enforced
there rather than by refusing to turn anything up.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

# What the platforms normalise towards. All three are close enough to one number
# that picking a separate target per platform would be false precision.
TARGET_LUFS = -14.0

# Highest sample the limiter will let through, in dBTP. Lossy encoders overshoot
# the waveform they are given, so the usual advice is to stop short of 0.
CEILING_DBTP = -1.0

# A file measured far below target is more likely to be misdetected than to
# genuinely need +25 dB, and a bad measurement should degrade quietly.
MAX_GAIN_DB = 12.0

# Below this the change is inaudible and the extra filter is not worth the
# re-encode risk.
MIN_GAIN_DB = 0.5


@dataclass(frozen=True)
class Loudness:
    """What one pass of ffmpeg's loudness meter found in a file."""

    integrated: float       # LUFS, the perceived average over the whole file
    true_peak: float        # dBTP, the highest reconstructed sample
    range: float            # LU, how far the quiet and loud parts sit apart

    @property
    def is_quiet(self) -> bool:
        return self.integrated < TARGET_LUFS - MIN_GAIN_DB

    @property
    def is_dynamic(self) -> bool:
        """
        Wide enough that a single gain will not sit right everywhere.

        Not acted on automatically. It is reported, because it is the honest
        answer to "why does clip 7 still sound quiet" and the fix for it is
        compression, which is a mastering decision rather than a repair.
        """
        return self.range > 14.0


def measure(source: str, timeout: float = 900.0) -> Optional[Loudness]:
    """
    Read the integrated loudness of a file.

    Returns None rather than raising if the file has no audio, ffmpeg is not
    installed, or the meter prints something unparseable. Nothing downstream
    needs this to succeed: without a measurement the audio is passed through
    untouched, which is exactly the behaviour this module replaced.
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", source,
             "-map", "0:a:0", "-af", "loudnorm=print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    return parse_loudnorm(proc.stderr)


def parse_loudnorm(text: str) -> Optional[Loudness]:
    """
    Pull the JSON block out of ffmpeg's stderr.

    The filter prints its report among the ordinary log lines rather than to
    stdout, and everything in it is a string, including the values that are
    numbers. A file too short to measure reports "-inf", which is not a level
    and is treated as no measurement at all.
    """
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        values = (float(data["input_i"]), float(data["input_tp"]),
                  float(data["input_lra"]))
    except (ValueError, KeyError, TypeError):
        return None

    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        return None
    return Loudness(*values)


def gain_for(measured: Optional[Loudness], target: float = TARGET_LUFS) -> float:
    """
    How far to move the level, in dB.

    Deliberately not clamped by the available peak headroom. See the note at the
    top of the module: the material that needs this most is the material with no
    headroom, and the limiter is what makes raising it safe.
    """
    if measured is None:
        return 0.0
    gain = target - measured.integrated
    return max(-MAX_GAIN_DB, min(MAX_GAIN_DB, gain))


def filter_chain(measured: Optional[Loudness],
                 target: float = TARGET_LUFS,
                 ceiling: float = CEILING_DBTP) -> Optional[str]:
    """
    The -af argument for a clip cut from this source, or None to leave it alone.

    The limiter runs whenever the result could reach the ceiling, which includes
    the case where no gain is applied but the source was already peaking. Its
    attack and release are slow enough not to audibly pump on speech and fast
    enough to catch a laugh or a slammed desk.
    """
    gain = gain_for(measured, target)
    peak_after = (measured.true_peak + gain) if measured else None

    steps = []
    if abs(gain) >= MIN_GAIN_DB:
        steps.append(f"volume={gain:.2f}dB")
    if peak_after is not None and peak_after > ceiling:
        steps.append(
            f"alimiter=limit={_amplitude(ceiling):.4f}:attack=5:release=50:level=disabled"
        )
    return ",".join(steps) or None


def _amplitude(dbtp: float) -> float:
    """dB relative to full scale, as the 0..1 amplitude alimiter wants."""
    return 10.0 ** (dbtp / 20.0)


def describe(measured: Optional[Loudness], target: float = TARGET_LUFS) -> str:
    """One line for the run log, in the units a person can act on."""
    if measured is None:
        return "audio level not measured, clips exported as-is"

    gain = gain_for(measured, target)
    direction = "up" if gain > 0 else "down"
    move = (f"turned {direction} {abs(gain):.1f} dB"
            if abs(gain) >= MIN_GAIN_DB else "already on target")
    note = ", wide dynamic range" if measured.is_dynamic else ""
    return (f"source measures {measured.integrated:.1f} LUFS, "
            f"peak {measured.true_peak:.1f} dBTP{note} - {move}")

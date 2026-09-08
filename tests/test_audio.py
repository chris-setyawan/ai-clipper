import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_clipper.video import audio


# What ffmpeg actually prints, log lines and all. The values are strings in the
# real output, which is the detail a hand-written fixture usually gets wrong.
FFMPEG_STDERR = """\
[Parsed_loudnorm_0 @ 0x55d1c]
{
	"input_i" : "-20.14",
	"input_tp" : "-0.83",
	"input_lra" : "9.60",
	"input_thresh" : "-30.42",
	"output_i" : "-24.01",
	"output_tp" : "-2.00",
	"output_lra" : "8.10",
	"output_thresh" : "-34.19",
	"normalization_type" : "dynamic",
	"target_offset" : "0.01"
}
"""

SILENT_STDERR = """\
[Parsed_loudnorm_0 @ 0x55d1c]
{
	"input_i" : "-inf",
	"input_tp" : "-120.00",
	"input_lra" : "0.00",
	"input_thresh" : "-inf"
}
"""


def test_parses_the_json_out_of_the_log():
    measured = audio.parse_loudnorm(FFMPEG_STDERR)
    assert measured is not None
    assert measured.integrated == -20.14
    assert measured.true_peak == -0.83
    assert measured.range == 9.60


def test_a_file_with_no_audible_content_is_not_a_measurement():
    assert audio.parse_loudnorm(SILENT_STDERR) is None


def test_output_with_no_report_at_all():
    assert audio.parse_loudnorm("ffmpeg version 6.0\nStream #0:0: Video\n") is None


def test_quiet_source_is_turned_up_towards_the_target():
    measured = audio.parse_loudnorm(FFMPEG_STDERR)
    gain = audio.gain_for(measured)
    assert math.isclose(gain, audio.TARGET_LUFS - (-20.14), abs_tol=0.01)
    assert gain > 0


def test_loud_source_is_turned_down():
    measured = audio.Loudness(integrated=-9.0, true_peak=-0.2, range=5.0)
    assert audio.gain_for(measured) < 0


def test_gain_is_clamped_so_a_bad_measurement_cannot_run_away():
    measured = audio.Loudness(integrated=-70.0, true_peak=-40.0, range=2.0)
    assert audio.gain_for(measured) == audio.MAX_GAIN_DB


def test_no_measurement_means_no_change():
    assert audio.gain_for(None) == 0.0
    assert audio.filter_chain(None) is None


def test_headroom_does_not_hold_the_gain_back():
    """
    The case the careful implementation gets wrong.

    This source is quiet on average and already peaking, which is what
    uncompressed conversation looks like. Clamping the gain to the available
    headroom would leave it alone; it needs both the gain and the limiter.
    """
    measured = audio.Loudness(integrated=-20.14, true_peak=-0.83, range=9.6)
    chain = audio.filter_chain(measured)
    assert chain is not None
    assert "volume=" in chain
    assert "alimiter" in chain
    assert chain.index("volume=") < chain.index("alimiter")


def test_a_source_already_on_target_is_left_alone():
    measured = audio.Loudness(integrated=-14.1, true_peak=-3.0, range=6.0)
    assert audio.filter_chain(measured) is None


def test_limiter_runs_even_without_gain_when_the_source_peaks():
    measured = audio.Loudness(integrated=-14.0, true_peak=-0.1, range=6.0)
    chain = audio.filter_chain(measured)
    assert chain is not None
    assert "volume=" not in chain
    assert "alimiter" in chain


def test_limiter_ceiling_is_expressed_as_an_amplitude():
    measured = audio.Loudness(integrated=-24.0, true_peak=-0.5, range=8.0)
    chain = audio.filter_chain(measured, ceiling=-1.0)
    # alimiter takes 0..1, not dB. -1 dBTP is about 0.891.
    assert "limit=0.891" in chain


def test_describe_says_which_way_and_how_far():
    measured = audio.parse_loudnorm(FFMPEG_STDERR)
    text = audio.describe(measured)
    assert "-20.1 LUFS" in text
    assert "turned up" in text
    assert "not measured" in audio.describe(None)

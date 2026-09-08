"""
Quick demo: load the sample transcript, rank hook candidates, print them out.

Run: python demo_scoring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_clipper.scoring.hook_scorer import HookScorer
from ai_clipper.scoring.transcript_io import load_json_transcript


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main():
    segments = load_json_transcript("data/sample_transcript.json")
    scorer = HookScorer()
    candidates = scorer.rank_candidates(segments, min_duration=15, max_duration=45, top_n=5)

    # print by score, highest first, so it reads like a "top picks" list
    candidates.sort(key=lambda c: c.score, reverse=True)

    print(f"Detected {len(candidates)} clip candidates from {len(segments)} transcript segments\n")
    for rank, cand in enumerate(candidates, start=1):
        print(f"#{rank}  HOT {cand.score}/10   [{format_time(cand.start)} - {format_time(cand.end)}]  ({cand.duration:.0f}s)")
        print(f"   why: {cand.explain()}")
        preview = cand.text[:140] + ("..." if len(cand.text) > 140 else "")
        print(f"   text: {preview}")
        print()


if __name__ == "__main__":
    main()

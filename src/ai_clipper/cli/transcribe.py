"""
Transcription - the only step that downloads a model.

The cloud sandbox I build in can't download Whisper model weights (its network
only reaches PyPI), so this step runs locally. It's self-contained: no need for
the rest of the project on your machine yet.

No system ffmpeg needed. faster-whisper installs PyAV, which carries its own
copy of the ffmpeg libraries, so the video is decoded in-process.

Setup (once):
    pip install faster-whisper

Run:
    ai-clipper-transcribe "podcast sample.mov"
    ai-clipper-transcribe "podcast sample.mov" --model medium
    ai-clipper-transcribe "podcast sample.mov" --lexicon lexicon.json

Output:
    transcript.json, written next to the video.

Models, roughly, on CPU: "base" is fastest and weakest, "small" is the default,
"medium" is noticeably better on Indonesian and about three times slower. If the
machine has an NVIDIA GPU, add --device cuda and "large-v3" becomes practical.

A lexicon fixes the words the model reliably gets wrong. It is used twice: fed
to the model as a hint before transcription, and applied as corrections after.
See lexicon.example.json.
"""

import argparse
import json
import os
import sys
import time


# A short, ordinary, fully punctuated Indonesian sentence. Its only job is to
# show Whisper what punctuated output looks like.
DEFAULT_PROMPT = "Ini adalah percakapan podcast. Mereka ngobrol santai, bertanya, dan bercerita."


def vocab_prompt_from_file(path):
    """
    Build Whisper's vocabulary hint straight from the lexicon JSON.

    Deliberately does not import the project: biasing the model before it runs
    is where most of the benefit is, and it should not be lost just because the
    package cannot be located.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    terms = list(data.get("vocabulary", [])) + list(data.get("exact", {}).values())
    seen, out = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    if not out:
        return None

    # a full sentence, not a bare list: Whisper imitates the prompt's style, and
    # an unpunctuated lowercase list makes it return an unpunctuated lowercase
    # transcript
    prefix = data.get("prompt_prefix", "Percakapan ini menyebut")
    return f"{prefix} {', '.join(out)}."


def load_lexicon(path):
    """
    Load the lexicon, or stop with an explanation.

    Failing loudly matters here: an earlier version skipped the lexicon with a
    quiet note when it could not find the package, and a whole transcription ran
    without the corrections anyone expected. If you asked for a lexicon and it
    cannot be used, that is an error, not a footnote.
    """
    if not path:
        return None

    if not os.path.exists(path):
        print(f"ERROR: lexicon file not found: {path}")
        sys.exit(1)

    from ..asr.corrections import Lexicon

    lex = Lexicon.load(path)
    print(f"Lexicon loaded from {path}: "
          f"{len(lex.exact)} exact fixes, {len(lex.vocabulary)} vocabulary terms")
    return lex


def main():
    ap = argparse.ArgumentParser(description="Transcribe a video or audio file.")
    ap.add_argument("media", help="path to the video or audio file, or a link")
    ap.add_argument("--model", default="small",
                    help="base | small | medium | large-v3 (default: small)")
    ap.add_argument("--language", default="id", help="language code (default: id)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    ap.add_argument("--lexicon", default=None, help="path to a lexicon.json")
    ap.add_argument("--out", default=None, help="output path (default: transcript.json beside the media)")
    args = ap.parse_args()

    if args.media.lower().startswith(("http://", "https://")):
        from ..input.fetch import resolve

        try:
            args.media = resolve(args.media)
        except Exception as err:
            print(f"ERROR: {err}")
            sys.exit(1)
    elif not os.path.exists(args.media):
        print(f"file not found: {args.media}")
        sys.exit(1)

    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    lex = load_lexicon(args.lexicon)
    if args.lexicon:
        # taken from the file directly, so the hint works even when the package
        # could not be found and corrections are being skipped
        prompt = vocab_prompt_from_file(args.lexicon)
        print(f"Vocabulary hint: {prompt}")
    else:
        # Whisper imitates the style of its prompt. With no prompt at all it may
        # return an entire hour with no punctuation, and every completeness rule
        # downstream reads punctuation - measured on one episode, the whole thing
        # went silent. A plain punctuated sentence costs nothing and asks for
        # punctuated output.
        prompt = DEFAULT_PROMPT
        print(f"Style hint: {prompt}")

    print("Decoding audio ...")
    audio = decode_audio(args.media, sampling_rate=16000)
    print(f"  {len(audio) / 16000:.0f}s of audio")

    compute = "float16" if args.device == "cuda" else "int8"
    print(f"Loading model '{args.model}' on {args.device} "
          f"(first run downloads it, please wait) ...")
    model = WhisperModel(args.model, device=args.device, compute_type=compute)

    print("Transcribing ...\n")
    t0 = time.time()
    segments_iter, info = model.transcribe(
        audio,
        language=args.language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        initial_prompt=prompt,
    )

    segments, words = [], []
    for seg in segments_iter:
        segments.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
            "speaker": None,
        })
        for w in (seg.words or []):
            words.append({
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "text": w.word.strip(),
            })
        print(f"  [{seg.start:7.1f}s] {seg.text.strip()[:70]}")

    out = {
        "source_file": os.path.basename(args.media),
        "language": info.language,
        "duration": round(float(info.duration), 2),
        "model": args.model,
        "segments": segments,
        "words": words,
    }

    if lex:
        from ai_clipper.asr.corrections import apply, diff_summary
        corrected = apply(out, lex)
        changes = diff_summary(out, corrected)
        out = corrected
        print(f"\nLexicon changed {len(changes)} words"
              + (f": {', '.join(f'{a} -> {b}' for a, b in changes[:8])}" if changes else ""))
    elif args.lexicon:
        print("\nNOTE: corrections were skipped (see the warning above); only the "
              "vocabulary hint was used.")

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.media)), "transcript.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"{len(out['segments'])} segments, {len(out['words'])} words")
    print(f"Written to: {out_path}")

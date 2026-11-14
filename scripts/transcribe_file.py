"""Transcribe a local audio file from the command line.

Usage:
    python scripts/transcribe_file.py samples/Test.mp3
    python scripts/transcribe_file.py samples/Test.mp3 --diarize --speakers 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.transcriber import Transcriber, format_speaker_turns  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", help="Path to an audio file")
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--diarize", action="store_true", help="Label speakers (needs HF_TOKEN)")
    parser.add_argument("--speakers", type=int, help="Exact number of speakers when diarizing")
    args = parser.parse_args()

    transcriber = Transcriber(
        model_name=args.model,
        device=settings.device,
        compute_type=settings.compute_type,
        diarize=args.diarize,
        hf_token=settings.hf_token,
    )

    result = transcriber.transcribe(
        args.audio,
        min_speakers=args.speakers,
        max_speakers=args.speakers,
    )

    print("\n" + "=" * 60)
    print(f"TRANSCRIPT  (language: {result['language']}, {result['duration']}s)")
    print("=" * 60)

    if args.diarize:
        for line in format_speaker_turns(result["segments"]):
            print(line)
    else:
        for seg in result["segments"]:
            print(f"[{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['text']}")

    print("=" * 60)


if __name__ == "__main__":
    main()

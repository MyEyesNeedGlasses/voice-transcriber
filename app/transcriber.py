import av
import numpy as np
import whisperx

SAMPLE_RATE = 16000


def load_audio(path):
    """Decode any audio/video file to 16 kHz mono float32.

    Uses PyAV (which bundles FFmpeg) instead of whisperx.load_audio, so no
    ffmpeg executable is needed on PATH.
    """
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []

    with av.open(path) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))

    for out in resampler.resample(None):
        chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        raise ValueError("No audio found in file")

    return np.concatenate(chunks).astype(np.float32) / 32768.0


class Transcriber:

    def __init__(
        self,
        model_name="small",
        device="cpu",
        compute_type="int8",
        diarize=False,
        hf_token=None,
    ):
        self.device = device
        self.diarize = diarize

        print(f"Loading WhisperX model ({model_name})...")
        self.model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type
        )

        # Alignment models are per-language; loaded on first use and cached
        self._align_models = {}
        self.diarize_model = None

        if diarize:
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN must be set when diarization is enabled."
                )

            # Imported lazily so pyannote is only loaded when needed
            from whisperx.diarize import DiarizationPipeline

            print("Loading speaker diarization model...")
            token = hf_token.strip()
            try:
                self.diarize_model = DiarizationPipeline(token=token, device=device)
            except TypeError:
                # Older whisperx releases name this argument use_auth_token
                self.diarize_model = DiarizationPipeline(use_auth_token=token, device=device)

    def _align(self, result, audio):
        """Add word-level timestamps, which speaker assignment needs."""
        language = result.get("language")
        if language not in self._align_models:
            try:
                self._align_models[language] = whisperx.load_align_model(
                    language_code=language, device=self.device
                )
            except Exception as exc:
                print(f"No alignment model for language {language!r}: {exc}")
                self._align_models[language] = None

        if self._align_models[language] is None:
            return result

        model, metadata = self._align_models[language]
        aligned = whisperx.align(
            result["segments"],
            model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        aligned["language"] = language
        return aligned

    def transcribe(self, audio_path, min_speakers=None, max_speakers=None):

        audio = load_audio(audio_path)
        duration = len(audio) / SAMPLE_RATE

        result = self.model.transcribe(audio)
        diarized = False

        if self.diarize_model is not None and result["segments"]:
            result = self._align(result, audio)
            diarize_segments = self.diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            result = whisperx.assign_word_speakers(
                diarize_segments,
                result
            )
            diarized = True

        if diarized:
            segments = speaker_turns(result)
        else:
            segments = [
                {
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip(),
                    "speaker": None,
                }
                for seg in result["segments"]
            ]

        return {
            "language": result.get("language"),
            "duration": round(duration, 2),
            "text": " ".join(s["text"] for s in segments if s["text"]),
            "segments": segments,
            "raw": result,
        }


def speaker_turns(result):
    """Merge consecutive words by the same speaker into one segment.

    WhisperX segments can span a speaker change, so turns are rebuilt from
    word-level speaker labels. Words without a label (e.g. numbers that could
    not be aligned) stay with the current speaker. Segments without word data
    fall back to their segment-level speaker.
    """
    turns = []

    def add(speaker, text, start, end):
        if turns and turns[-1]["speaker"] == speaker:
            turn = turns[-1]
            turn["text"] = f"{turn['text']} {text}"
            if start is not None and turn["start"] is None:
                turn["start"] = start
            if end is not None:
                turn["end"] = end
        else:
            turns.append({"speaker": speaker, "text": text, "start": start, "end": end})

    for segment in result["segments"]:
        words = segment.get("words") or []
        if not words:
            text = segment.get("text", "").strip()
            if text:
                add(segment.get("speaker"), text, segment.get("start"), segment.get("end"))
            continue

        for word in words:
            text = word.get("word", "").strip()
            if not text:
                continue
            speaker = word.get("speaker")
            if speaker is None:
                speaker = turns[-1]["speaker"] if turns else segment.get("speaker")
            add(speaker, text, word.get("start"), word.get("end"))

    # Fill any missing timestamps from neighbours so the schema stays valid
    for i, turn in enumerate(turns):
        if turn["start"] is None:
            turn["start"] = turns[i - 1]["end"] if i else 0.0
        if turn["end"] is None:
            turn["end"] = turn["start"]
        turn["start"] = round(turn["start"], 2)
        turn["end"] = round(turn["end"], 2)

    return turns


def format_speaker_turns(segments):
    """Render segments as "SPEAKER: text" lines."""
    return [f"{seg['speaker'] or 'UNKNOWN'}: {seg['text']}" for seg in segments]

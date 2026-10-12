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

        self.diarize_model = None

        if diarize:
            if not hf_token:
                raise ValueError(
                    "HF_TOKEN must be set when diarization is enabled."
                )

            # Imported lazily so pyannote is only loaded when needed
            from whisperx.diarize import DiarizationPipeline

            print("Loading speaker diarization model...")
            self.diarize_model = DiarizationPipeline(
                token=hf_token.strip(),
                device=device
            )

    def transcribe(self, audio_path, min_speakers=None, max_speakers=None):

        audio = load_audio(audio_path)
        duration = len(audio) / SAMPLE_RATE

        result = self.model.transcribe(audio)

        if self.diarize_model is not None:
            diarize_segments = self.diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            result = whisperx.assign_word_speakers(
                diarize_segments,
                result
            )

        segments = [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
                "speaker": seg.get("speaker"),
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


def format_speaker_turns(result):
    """Group consecutive words by speaker into "SPEAKER: text" lines."""

    lines = []
    current_speaker = None
    current_text = []

    for segment in result["segments"]:

        for word in segment.get("words", []):

            speaker = word.get("speaker", "UNKNOWN")
            text = word.get("word", "").strip()

            if not text:
                continue

            if current_speaker is None:
                current_speaker = speaker

            if speaker != current_speaker:
                lines.append(f"{current_speaker}: {' '.join(current_text)}")
                current_speaker = speaker
                current_text = []

            current_text.append(text)

    if current_text:
        lines.append(f"{current_speaker}: {' '.join(current_text)}")

    return lines

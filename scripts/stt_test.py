import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MODEL_SIZE = sys.argv[1] if len(sys.argv) > 1 else "base.en"


def record(seconds: int) -> np.ndarray:
    print(f"Recording for {seconds} seconds. Speak now...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def main() -> None:
    start = time.perf_counter()
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", download_root="models/whisper")
    print(f"[model {MODEL_SIZE} loaded in {time.perf_counter() - start:.2f}s]")

    audio = record(SECONDS)
    peak = float(np.abs(audio).max())
    print(f"[peak level {peak:.3f}]")
    if peak < 0.01:
        print("Almost silent. Check that VS Code has microphone permission.")
        return

    start = time.perf_counter()
    segments, _info = model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments)
    elapsed = time.perf_counter() - start
    print(f"[transcribed {SECONDS}s of audio in {elapsed:.2f}s]")
    print(f"You said: {text!r}")


if __name__ == "__main__":
    main()
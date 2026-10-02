import time

import numpy as np
import sounddevice as sd
from piper import PiperVoice

VOICE_PATH = "models/piper/en_US-lessac-medium.onnx"
SENTENCES = [
    "Here are the open times on Tuesday, October 6: 9 a.m., 10 a.m., or 11 a.m.",
    "Shall I book 11 a.m. on Tuesday, October 6 for Anvitha Reddy?",
    "You're booked. Your confirmation code is 6 5 0 C C 1.",
]


def main() -> None:
    start = time.perf_counter()
    voice = PiperVoice.load(VOICE_PATH)
    print(f"[voice loaded in {time.perf_counter() - start:.2f}s]")

    for text in SENTENCES:
        start = time.perf_counter()
        first_audio = None
        sample_rate = None
        chunks = []

        for chunk in voice.synthesize(text):
            if first_audio is None:
                first_audio = time.perf_counter() - start
            sample_rate = chunk.sample_rate
            chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))

        total = time.perf_counter() - start
        audio = np.concatenate(chunks)
        duration = len(audio) / sample_rate

        print(f"[first audio {first_audio:.2f}s | {duration:.1f}s of speech made in {total:.2f}s | {sample_rate} Hz]")
        print(f"Agent: {text}")
        sd.play(audio, sample_rate)
        sd.wait()


if __name__ == "__main__":
    main()
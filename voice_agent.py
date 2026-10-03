import os
import queue
import re
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice

from agent import Session, log, warm_up

SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK = SAMPLE_RATE * BLOCK_MS // 1000
SILENCE_MS_TO_END = 700
MAX_UTTERANCE_S = 15
PRE_ROLL_BLOCKS = 10
WHISPER_MODEL = "base.en"
VOICE_PATH = "models/piper/en_US-lessac-medium.onnx"
GREETING = "Hi, I can help you book an appointment. Which day works for you?"
BYE_RE = re.compile(r"\b(bye|goodbye)\b", re.IGNORECASE)
INPUT_DEVICE = os.environ.get("INPUT_DEVICE")
OUTPUT_DEVICE = os.environ.get("OUTPUT_DEVICE")


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block**2)))


class Microphone:
    """One input stream for the whole session, so headsets don't switch modes every turn."""

    def __init__(self, device: str | None) -> None:
        self.blocks: queue.Queue = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK,
            device=device,
            callback=self._on_audio,
        )
        self.stream.start()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        self.blocks.put(indata[:, 0].copy())

    def drain(self) -> None:
        """Throw away audio captured while the agent was speaking."""
        while True:
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                return

    def read(self) -> np.ndarray:
        return self.blocks.get()

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


def calibrate(mic: Microphone, seconds: float = 1.0, skip: float = 0.3) -> float:
    time.sleep(skip)
    mic.drain()
    levels = [rms(mic.read()) for _ in range(int(seconds * 1000 / BLOCK_MS))]
    noise = float(np.median(levels))
    threshold = max(noise * 3, 0.01)
    log(f"noise floor {noise:.4f}, speech threshold {threshold:.4f}")
    return threshold


def listen(mic: Microphone, threshold: float) -> np.ndarray:
    mic.drain()
    pre_roll: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    speaking = False
    silent_ms = 0

    while True:
        block = mic.read()
        loud = rms(block) > threshold

        if not speaking:
            pre_roll = (pre_roll + [block])[-PRE_ROLL_BLOCKS:]
            if loud:
                speaking = True
                frames = list(pre_roll)
            continue

        frames.append(block)
        silent_ms = 0 if loud else silent_ms + BLOCK_MS
        too_long = len(frames) * BLOCK_MS >= MAX_UTTERANCE_S * 1000
        if silent_ms >= SILENCE_MS_TO_END or too_long:
            return np.concatenate(frames)


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    segments, _info = model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def speak(voice: PiperVoice, text: str, turn_start: float | None = None) -> None:
    chunks = []
    sample_rate = None
    for chunk in voice.synthesize(text):
        chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        sample_rate = chunk.sample_rate
    if not chunks:
        return
    if turn_start is not None:
        log(f"turn latency {time.perf_counter() - turn_start:.2f}s (end of speech -> first audio)")
    sd.play(np.concatenate(chunks), sample_rate, device=OUTPUT_DEVICE)
    sd.wait()


def main() -> None:
    print("Loading models...")
    stt = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", download_root="models/whisper")
    voice = PiperVoice.load(VOICE_PATH)
    for _ in voice.synthesize("Warming up."):
        pass
    warm_up()
    session = Session()

    mic = Microphone(INPUT_DEVICE)
    log(f"input device: {INPUT_DEVICE or 'system default'}, output device: {OUTPUT_DEVICE or 'system default'}")
    try:
        print("Stay quiet for one second while I measure background noise...")
        threshold = calibrate(mic)
        print("Ready. Say 'goodbye' or press Ctrl+C to stop.\n")

        print(f"Agent: {GREETING}")
        speak(voice, session.say(GREETING))

        while True:
            audio = listen(mic, threshold)
            turn_start = time.perf_counter()

            text = transcribe(stt, audio)
            log(f"stt {time.perf_counter() - turn_start:.2f}s ({len(audio) / SAMPLE_RATE:.1f}s of audio)")
            if not text:
                log("heard nothing, listening again")
                continue
            print(f"You: {text}")

            if BYE_RE.search(text):
                print("Agent: Goodbye!")
                speak(voice, "Goodbye!", turn_start)
                break

            reply = session.reply(text)
            print(f"Agent: {reply}")
            speak(voice, reply, turn_start)
    except KeyboardInterrupt:
        print()
    finally:
        mic.close()


if __name__ == "__main__":
    main()
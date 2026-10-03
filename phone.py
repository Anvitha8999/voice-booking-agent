import asyncio
import base64
import json
import math
import os
import re
import time
from xml.sax.saxutils import quoteattr

import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from faster_whisper import WhisperModel
from piper import PiperVoice
from scipy.signal import resample_poly

from agent import Session, log, warm_up
from audio import mulaw_decode, mulaw_encode

router = APIRouter()

PHONE_RATE = 8000
FRAME_MS = 20
SILENCE_MS_TO_END = 700
MAX_UTTERANCE_S = 15
PRE_ROLL_FRAMES = 15
SPEECH_THRESHOLD = float(os.environ.get("PHONE_VAD_THRESHOLD", "0.02"))
SEND_CHUNK_BYTES = 8000
GREETING = "Hi, I can help you book an appointment. Which day works for you?"
GOODBYE = "Thanks for calling. Goodbye!"
BYE_RE = re.compile(r"\b(bye|goodbye)\b", re.IGNORECASE)

log("loading speech models for phone calls...")
stt = WhisperModel("base.en", device="cpu", compute_type="int8", download_root="models/whisper")
voice = PiperVoice.load("models/piper/en_US-lessac-medium.onnx")
for _ in voice.synthesize("Warming up."):
    pass
warm_up()
log("phone models ready")


def mask(number: str | None) -> str:
    return f"***{number[-4:]}" if number else "unknown"


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    g = math.gcd(from_rate, to_rate)
    return resample_poly(audio, to_rate // g, from_rate // g).astype(np.float32)


def transcribe(pcm: np.ndarray) -> str:
    audio = resample(pcm.astype(np.float32) / 32768, PHONE_RATE, 16000)
    segments, _info = stt.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def synthesize_mulaw(text: str) -> bytes:
    chunks = []
    sample_rate = PHONE_RATE
    for chunk in voice.synthesize(text):
        chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        sample_rate = chunk.sample_rate
    if not chunks:
        return b""
    audio = np.concatenate(chunks).astype(np.float32) / 32768
    audio = resample(audio, sample_rate, PHONE_RATE)
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return mulaw_encode(pcm)


@router.post("/voice")
async def voice_webhook(request: Request):
    form = dict(await request.form())
    caller = str(form.get("From", ""))
    log(f"incoming call {form.get('CallSid')} from {mask(caller)}")

    host = request.headers["host"]
    stream_url = quoteattr(f"wss://{host}/media-stream")
    log(f"telling Twilio to stream to {stream_url}")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url={stream_url}>
      <Parameter name="caller" value={quoteattr(caller)} />
    </Stream>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


class Call:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.stream_sid: str | None = None
        self.session: Session | None = None
        self.state = "thinking"
        self.hangup_after_speaking = False
        self.tasks: set[asyncio.Task] = set()
        self.reset_vad()

    def reset_vad(self) -> None:
        self.pre_roll: list[np.ndarray] = []
        self.frames: list[np.ndarray] = []
        self.speaking = False
        self.silent_ms = 0

    def run(self, coro) -> None:
        """Run a coroutine in the background, keeping a reference and logging failures."""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and task.exception():
            log(f"turn failed: {task.exception()!r}")
            self.state = "listening"

    def feed(self, pcm: np.ndarray) -> np.ndarray | None:
        """Feed one 20 ms frame; return the full utterance once the caller stops talking."""
        level = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768) ** 2)))
        loud = level > SPEECH_THRESHOLD

        if not self.speaking:
            self.pre_roll = (self.pre_roll + [pcm])[-PRE_ROLL_FRAMES:]
            if loud:
                self.speaking = True
                self.frames = list(self.pre_roll)
            return None

        self.frames.append(pcm)
        self.silent_ms = 0 if loud else self.silent_ms + FRAME_MS
        too_long = len(self.frames) * FRAME_MS >= MAX_UTTERANCE_S * 1000
        if self.silent_ms >= SILENCE_MS_TO_END or too_long:
            utterance = np.concatenate(self.frames)
            self.reset_vad()
            return utterance
        return None

    async def send_audio(self, mulaw: bytes) -> None:
        for i in range(0, len(mulaw), SEND_CHUNK_BYTES):
            payload = base64.b64encode(mulaw[i : i + SEND_CHUNK_BYTES]).decode()
            await self.ws.send_text(
                json.dumps({"event": "media", "streamSid": self.stream_sid, "media": {"payload": payload}})
            )
        await self.ws.send_text(
            json.dumps({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": "end_of_reply"}})
        )

    async def say(self, text: str, turn_start: float | None = None, hangup: bool = False) -> None:
        mulaw = await asyncio.to_thread(synthesize_mulaw, text)
        if turn_start is not None:
            log(f"turn latency {time.perf_counter() - turn_start:.2f}s (end of speech -> audio sent)")
        self.state = "speaking"
        self.hangup_after_speaking = hangup
        await self.send_audio(mulaw)
        log(f"sent {len(mulaw) / PHONE_RATE:.1f}s of audio")

    async def handle_turn(self, pcm: np.ndarray) -> None:
        turn_start = time.perf_counter()
        text = await asyncio.to_thread(transcribe, pcm)
        log(f"stt {time.perf_counter() - turn_start:.2f}s ({len(pcm) / PHONE_RATE:.1f}s of audio)")
        if not text:
            log("heard nothing, listening again")
            self.state = "listening"
            return

        log(f"caller: {text}")
        if BYE_RE.search(text):
            await self.say(GOODBYE, turn_start, hangup=True)
            return

        reply = await asyncio.to_thread(self.session.reply, text)
        log(f"agent: {reply}")
        await self.say(reply, turn_start)


@router.websocket("/media-stream")
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()
    log("media stream websocket connected")
    call = Call(ws)
    media_frames = 0
    try:
        while True:
            raw = await ws.receive_text()
            message = json.loads(raw)
            event = message.get("event")
            if event != "media":
                log(f"twilio event: {event}")

            if event == "start":
                start = message["start"]
                call.stream_sid = start["streamSid"]
                caller = start.get("customParameters", {}).get("caller")
                call.session = Session(caller_phone=caller or None)
                log(f"stream started {call.stream_sid} for caller {mask(caller)}, format {start.get('mediaFormat')}")
                call.run(call.say(call.session.say(GREETING)))

            elif event == "media":
                media_frames += 1
                if media_frames == 1:
                    log("first audio frame received")
                if call.state != "listening":
                    continue
                pcm = mulaw_decode(base64.b64decode(message["media"]["payload"]))
                utterance = call.feed(pcm)
                if utterance is not None:
                    call.state = "thinking"
                    call.run(call.handle_turn(utterance))

            elif event == "mark":
                if call.hangup_after_speaking:
                    log("goodbye played, hanging up")
                    await ws.close()
                    break
                call.state = "listening"
                call.reset_vad()

            elif event == "stop":
                log("caller hung up")
                break
    except WebSocketDisconnect as e:
        log(f"websocket disconnected (code {e.code}) after {media_frames} audio frames")
    except Exception as e:
        log(f"media stream error: {e!r}")
        raise
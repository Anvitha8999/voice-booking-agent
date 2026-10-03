import asyncio
import json
import sys

import websockets

URL = sys.argv[1]


async def main() -> None:
    async with websockets.connect(URL) as ws:
        print("connected")
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({
            "event": "start",
            "start": {
                "streamSid": "MZTEST",
                "callSid": "CATEST",
                "customParameters": {"caller": "+15550001234"},
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            },
        }))
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            payload = message.get("media", {}).get("payload", "")
            print(f"received {message['event']} ({len(payload)} base64 chars)")
            if message["event"] == "mark":
                print("greeting fully received")
                return


if __name__ == "__main__":
    asyncio.run(main())
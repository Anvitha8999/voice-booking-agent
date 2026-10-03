import numpy as np

BIAS = 0x84
CLIP = 32635


def mulaw_decode(data: bytes) -> np.ndarray:
    """G.711 mu-law bytes -> 16-bit PCM samples."""
    u = (~np.frombuffer(data, dtype=np.uint8)).astype(np.int32)
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + BIAS) << exponent) - BIAS
    return np.where(sign != 0, -sample, sample).astype(np.int16)


def mulaw_encode(pcm: np.ndarray) -> bytes:
    """16-bit PCM samples -> G.711 mu-law bytes."""
    s = pcm.astype(np.int32)
    sign = (s < 0).astype(np.int32) << 7
    magnitude = np.minimum(np.abs(s), CLIP) + BIAS
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return encoded.astype(np.uint8).tobytes()
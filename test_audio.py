import numpy as np

from audio import mulaw_decode, mulaw_encode


def test_silence_encodes_to_0xff():
    assert mulaw_encode(np.zeros(4, dtype=np.int16)) == b"\xff\xff\xff\xff"
    assert np.all(mulaw_decode(b"\xff\xff") == 0)


def test_roundtrip_error_is_within_mulaw_step():
    x = np.arange(-32768, 32768, 7, dtype=np.int32)
    decoded = mulaw_decode(mulaw_encode(x.astype(np.int16))).astype(np.int32)
    clipped = np.clip(x, -32635, 32635)
    tolerance = (np.abs(clipped) + 0x84) // 32 + 2
    assert np.all(np.abs(decoded - clipped) <= tolerance)


def test_sign_is_preserved():
    pcm = np.array([1000, -1000, 20000, -20000], dtype=np.int16)
    decoded = mulaw_decode(mulaw_encode(pcm))
    assert np.all(np.sign(decoded) == np.sign(pcm))
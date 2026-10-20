"""Small synthetic videos whose frames encode their own sequence number, for decoding tests."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

W, H = 128, 64
BITS = 8


def pattern(i: int) -> np.ndarray:
    """RGB frame showing ``i`` as eight black/white blocks, plus a marker in the top-left corner."""
    img = np.zeros((H, W, 3), np.uint8)
    for b in range(BITS):
        if i >> b & 1:
            img[16:48, b * 16:(b + 1) * 16] = 255
    img[:8, :8] = (255, 0, 0)
    return img


def read_index(img_bgr: np.ndarray) -> int:
    """Inverse of ``pattern`` for an upright, full-size BGR frame."""
    return sum(1 << b for b in range(BITS) if img_bgr[32, b * 16 + 8].mean() > 128)


def make_video(path: Path, times: Sequence[float], codec: str = "mpeg4", fmt: Optional[str] = None,
               rotation: int = 0, gop: int = 4, options: Optional[dict] = None) -> Path:
    """Encode one frame per timestamp (seconds) with millisecond timestamps."""
    import av
    with av.open(str(path), "w", format=fmt) as c:
        s = c.add_stream(codec, rate=1000)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.codec_context.gop_size = gop
        if options:
            s.options = options
        if rotation:
            s.set_display_rotation(rotation)
        for i, t in enumerate(times):
            f = av.VideoFrame.from_ndarray(pattern(i), format="rgb24")
            f.pts, f.time_base = round(t * 1000), Fraction(1, 1000)
            for pk in s.encode(f):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return Path(path)


def cfr_times(n: int, fps: float) -> list[float]:
    return [round(i / fps, 3) for i in range(n)]


VFR = [0.0, 0.1, 0.2, 0.5, 0.55, 1.6, 1.7, 1.8, 1.9, 2.0]

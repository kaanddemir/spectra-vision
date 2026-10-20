"""Small validated cache records shared by cache and streaming tests."""
from spectra.engine.cache import perception_key
from spectra.engine.contracts import (ComponentInfo, FrameRecord, OutputStatus, PerceptionResult, Provenance, Role,
                                      RunConfig, TimingReport)
from tests.engine.fakes import video

DET = ComponentInfo(role=Role.DETECTOR, name="det", version="1", params={"input_px": 640})
SEG = ComponentInfo(role=Role.SEGMENTER, name="seg", version="1")


def provenance(*components):
    return Provenance(engine_version="0", decoder={"pyav": "19.0.1"}, components=components)


def result(components=(DET,), config=RunConfig(interval_s=0.2), refs=None):
    frames = []
    for i in range(3):
        status = {c.role: OutputStatus.OK for c in components}
        extra = {}
        if Role.DETECTOR in status:
            extra["detections"] = ()
        if Role.SEGMENTER in status:
            extra["segmentation"] = refs[i]
        frames.append(FrameRecord(index=i, t=i * 0.2, status=status, **extra))
    return PerceptionResult(video=video(), config=config, provenance=provenance(*components),
                            timing=TimingReport(frames_decoded=3, frames_analysed=3), frames=tuple(frames))


def store(cache, r):
    key = perception_key(r.video, r.config, r.provenance)
    cache.begin(key).commit(r)
    return key

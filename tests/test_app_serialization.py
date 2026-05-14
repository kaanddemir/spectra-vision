from spectra.app import _serialize_result


def test_serialize_result_includes_performance_logs():
    summary = {
        "processed_frames": 12,
        "elapsed_sec": 0.123,
        "effective_fps": 97.56,
    }
    result = {
        "fps": 30.0,
        "frame_count": 12,
        "processed_frames": 12,
        "frames": [],
        "events": [],
        "peak_event": None,
        "performance_summary": summary,
        "performance_logs": ["[FRAME    0] preprocess=1ms"],
    }

    serialized = _serialize_result(result, elapsed_sec=0.1234, source_name="sample.mp4")

    assert serialized["payload"]["performance_summary"] == summary
    assert serialized["payload"]["performance_logs"] == ["[FRAME    0] preprocess=1ms"]

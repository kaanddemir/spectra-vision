import pytest

from spectra.analysis.video import (
    _build_performance_summary,
    _empty_performance_stats,
    _format_performance_summary,
    _record_stage,
)


def test_performance_summary_reports_stage_stats_and_bottleneck():
    stats = _empty_performance_stats()
    for elapsed_ms in (10, 20, 30):
        _record_stage(stats, "preprocess", elapsed_ms / 1000.0)
    for elapsed_ms in (100, 200):
        _record_stage(stats, "lane", elapsed_ms / 1000.0)
    _record_stage(stats, "lane", 0.005, active=False)
    for elapsed_ms in (5, 10, 15):
        _record_stage(stats, "flow", elapsed_ms / 1000.0)
    _record_stage(stats, "depth", 0.08)
    _record_stage(stats, "depth", 0.002, active=False)
    for elapsed_ms in (150, 250, 350):
        _record_stage(stats, "yolo", elapsed_ms / 1000.0)

    summary = _build_performance_summary(
        stats,
        processed_frames=3,
        elapsed_sec=1.5,
        lane_every=5,
        flow_every=1,
        depth_every=10,
        adaptive_depth=True,
        detect_every=3,
        depth_refresh={
            "runs": 2,
            "skips": 1,
            "initial_runs": 1,
            "periodic_runs": 1,
            "motion_triggered_runs": 0,
            "cooldown_frames": 5,
        },
    )

    assert summary["processed_frames"] == 3
    assert summary["effective_fps"] == pytest.approx(2.0)
    assert summary["depth_refresh"]["runs"] == 2
    assert summary["depth_refresh"]["skips"] == 1
    assert summary["depth_refresh"]["effective_interval_frames"] == pytest.approx(1.5)
    assert summary["bottleneck"]["stage"] == "yolo"
    assert summary["stages"]["preprocess"]["active"]["avg_ms"] == pytest.approx(20.0)
    assert summary["stages"]["preprocess"]["active"]["p95_ms"] == pytest.approx(30.0)
    assert summary["stages"]["preprocess"]["active"]["max_ms"] == pytest.approx(30.0)
    assert summary["stages"]["lane"]["active"]["avg_ms"] == pytest.approx(150.0)
    assert summary["stages"]["lane"]["frame"]["avg_ms"] == pytest.approx(101.6666667)

    lines = _format_performance_summary(summary)
    assert lines[0] == "SUMMARY"
    assert any("Bottleneck: yolo" in line for line in lines)
    assert any(
        "Depth refresh: runs=2 skips=1 initial=1 periodic=1 motion=0 cooldown=5 effective_interval=1.5f" in line
        for line in lines
    )
    assert any("Sampling: lane_every=5 depth_every=10 adaptive_depth=on detect_every=3 flow_every=1" in line for line in lines)

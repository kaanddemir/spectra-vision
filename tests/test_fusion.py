"""Unit tests for fusion.py — zone_regions and compute_quick_risk."""

import pytest
import numpy as np

from zone_risk.pipeline.fusion import zone_regions, compute_quick_risk, ZoneRegion
from zone_risk.vision.optical_flow import FlowResult


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_flow_result(height=100, width=100, magnitude=0.5, divergence=0.3):
    return FlowResult(
        flow=np.zeros((height, width, 2), dtype=np.float32),
        magnitude_norm=np.full((height, width), magnitude, dtype=np.float32),
        divergence_norm=np.full((height, width), divergence, dtype=np.float32),
    )


# ── zone_regions ─────────────────────────────────────────────────────────────

class TestZoneRegions:
    def test_returns_three_zones(self):
        regions = list(zone_regions(width=300, height=100))
        assert len(regions) == 3

    def test_zone_labels(self):
        labels = {r.label for r in zone_regions(300, 100)}
        assert labels == {"left zone", "center zone", "right zone"}

    def test_zones_cover_full_width(self):
        regions = list(zone_regions(width=300, height=100))
        x_starts = sorted(r.bbox[0] for r in regions)
        x_ends = sorted(r.bbox[2] for r in regions)
        assert x_starts[0] == 0
        assert x_ends[-1] == 300

    def test_zones_do_not_overlap(self):
        regions = sorted(zone_regions(300, 100), key=lambda r: r.bbox[0])
        for i in range(len(regions) - 1):
            assert regions[i].bbox[2] == regions[i + 1].bbox[0]

    def test_center_zone_is_twice_side_zones(self):
        regions = list(zone_regions(width=400, height=100))
        widths = {r.label: r.bbox[2] - r.bbox[0] for r in regions}
        assert widths["center zone"] == widths["left zone"] * 2
        assert widths["center zone"] == widths["right zone"] * 2

    def test_unique_ids(self):
        ids = [r.id for r in zone_regions(300, 100)]
        assert len(ids) == len(set(ids))


# ── compute_quick_risk ────────────────────────────────────────────────────────

class TestComputeQuickRisk:
    def test_returns_float(self):
        flow = make_flow_result()
        result = compute_quick_risk(flow, width=100, height=100)
        assert isinstance(result, float)

    def test_zero_motion_gives_low_risk(self):
        flow = make_flow_result(magnitude=0.0, divergence=0.0)
        risk = compute_quick_risk(flow, width=100, height=100)
        assert risk < 0.1

    def test_high_motion_gives_high_risk(self):
        flow = make_flow_result(magnitude=1.0, divergence=1.0)
        risk = compute_quick_risk(flow, width=100, height=100)
        assert risk > 0.5

    def test_result_is_bounded(self):
        flow = make_flow_result(magnitude=1.0, divergence=1.0)
        risk = compute_quick_risk(flow, width=100, height=100)
        assert 0.0 <= risk <= 1.0

    def test_center_motion_scores_higher_than_edge_motion(self):
        h, w = 100, 300

        center_flow_arr = np.zeros((h, w, 2), dtype=np.float32)
        center_mag = np.zeros((h, w), dtype=np.float32)
        center_mag[:, w // 3: 2 * w // 3] = 1.0  # only center has motion
        center_flow = FlowResult(
            flow=center_flow_arr,
            magnitude_norm=center_mag,
            divergence_norm=np.zeros((h, w), dtype=np.float32),
        )

        edge_flow_arr = np.zeros((h, w, 2), dtype=np.float32)
        edge_mag = np.zeros((h, w), dtype=np.float32)
        edge_mag[:, :w // 6] = 1.0   # only far-left edge
        edge_mag[:, 5 * w // 6:] = 1.0
        edge_flow = FlowResult(
            flow=edge_flow_arr,
            magnitude_norm=edge_mag,
            divergence_norm=np.zeros((h, w), dtype=np.float32),
        )

        center_risk = compute_quick_risk(center_flow, width=w, height=h)
        edge_risk = compute_quick_risk(edge_flow, width=w, height=h)
        assert center_risk > edge_risk

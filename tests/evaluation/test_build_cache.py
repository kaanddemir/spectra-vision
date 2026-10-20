import cv2
import numpy as np
import pytest

from evaluation.build_cache import global_motion


@pytest.fixture
def texture():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((360, 640)) * 255).astype(np.uint8), (5, 5), 0)
    return img


def test_recovers_translation(texture):
    shifted = np.roll(np.roll(texture, 6, axis=1), -4, axis=0)
    tx, ty, rot, log_s, inliers, _ = global_motion(texture, shifted)
    assert tx == pytest.approx(6, abs=0.5) and ty == pytest.approx(-4, abs=0.5)
    assert abs(rot) < 0.01 and abs(log_s) < 0.01 and inliers > 0.5


def test_recovers_rotation(texture):
    m = cv2.getRotationMatrix2D((320, 180), 3.0, 1.0)
    rotated = cv2.warpAffine(texture, m, (640, 360), borderMode=cv2.BORDER_REFLECT)
    _, _, rot, _, _, _ = global_motion(texture, rotated)
    assert abs(rot) == pytest.approx(np.deg2rad(3.0), abs=0.01)


def test_featureless_frames_report_unknown_motion():
    flat = np.full((360, 640), 128, np.uint8)
    tx, ty, rot, log_s, inliers, photometric = global_motion(flat, flat)
    assert np.isnan(tx) and inliers == 0.0 and photometric == 0.0


@pytest.mark.parametrize("adapter", ["dfine", "rfdetr-nano", "rfdetr-small"])
def test_detector_checkpoint_hash_is_saved_in_new_manifest(tmp_path, monkeypatch, adapter):
    import hashlib
    import json

    from evaluation.build_cache import detector_model_info
    from evaluation.detectors.dfine import DFine
    from evaluation.detectors import rfdetr
    from evaluation.perception_cache import PerceptionCache

    # Real adapter path selection, without loading a model or decoding media.
    checkpoint = tmp_path / ("custom.pth" if adapter == "dfine" else f"rf-detr-{adapter.split('-')[1]}.pth")
    checkpoint.write_bytes(b"local checkpoint fixture")
    if adapter == "dfine":
        det = DFine(weights=checkpoint)
    else:
        monkeypatch.setattr(rfdetr, "WEIGHTS", tmp_path)
        det = rfdetr.RfDetr(size=adapter.split('-')[1])
    info = detector_model_info(det, {"threads": 2})
    cache = PerceptionCache(media_sha256="fixture", frame_size=(10, 10),
                            timestamps=np.array([0.0]), detections=np.empty((0, 8)), models=(info,))
    destination = tmp_path / "new-cache"
    cache.save(destination)
    model = json.loads((destination / "manifest.json").read_text())["models"][0]
    assert model["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert PerceptionCache.load(destination).models[0] == info


def test_unknown_checkpoint_stays_unknown():
    from types import SimpleNamespace
    from evaluation.build_cache import detector_model_info

    assert detector_model_info(SimpleNamespace(name="custom"), {}).sha256 is None


def test_missing_declared_checkpoint_is_refused(tmp_path):
    from types import SimpleNamespace
    from evaluation.build_cache import detector_model_info

    with pytest.raises(FileNotFoundError):
        detector_model_info(SimpleNamespace(name="custom", weights=tmp_path / "missing.pth"), {})

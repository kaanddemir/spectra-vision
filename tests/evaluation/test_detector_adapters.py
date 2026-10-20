import numpy as np
import pytest

from evaluation.detectors import VEHICLE_CLASSES
from evaluation.detectors.rfdetr import to_contiguous


def test_rfdetr_coco_category_ids_map_to_contiguous_zero_based_ids():
    # person, bicycle, car, motorcycle, bus, truck, toothbrush
    assert list(to_contiguous(np.array([1, 2, 3, 4, 6, 8, 90]))) == [0, 1, 2, 3, 5, 7, 79]
    assert {2: "car", 7: "truck"}.items() <= VEHICLE_CLASSES.items()


def test_unknown_coco_category_ids_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        to_contiguous(np.array([12]))  # unused COCO id

"""Tests for the TTC cue-agreement reliability signal.

Agreement is derived purely from the already-computed TTC components and may
only *temper* trust (never fabricate risk). These tests pin that behaviour.
"""

from spectra.analysis.risk import TtcComponent, ttc_agreement


def _c(value, conf=0.5):
    return TtcComponent("x", value, conf)


def test_agreement_high_when_cues_tight():
    agree = ttc_agreement([_c(1.9), _c(2.0), _c(2.1)])
    assert agree > 0.85


def test_agreement_low_when_cues_diverge():
    agree = ttc_agreement([_c(0.5), _c(3.0), _c(8.0)])
    assert agree < 0.2


def test_agreement_neutral_with_fewer_than_two_cues():
    # Nothing to disagree about → never penalise (eta term is already 0 anyway).
    assert ttc_agreement([_c(2.0)]) == 1.0
    assert ttc_agreement([]) == 1.0
    assert ttc_agreement([_c(None), _c(2.0)]) == 1.0


def test_agreement_ignores_zero_confidence_components():
    # A cue with no confidence shouldn't count toward agreement.
    assert ttc_agreement([_c(2.0, 0.0), _c(2.0, 0.5)]) == 1.0

from evaluation.sampling import select

ROWS = [{"id": f"{i:03d}", "target": str(i % 2), "light": ["day", "night", "dusk"][i % 3]} for i in range(60)]


def test_selection_is_deterministic_and_balanced_by_label():
    a = select(ROWS, "id", "target", ["light"], 6, seed=1)
    assert a == select(ROWS, "id", "target", ["light"], 6, seed=1)
    assert sum(r["target"] == "1" for r in a) == 6 and sum(r["target"] == "0" for r in a) == 6


def test_selection_covers_every_stratum_when_possible():
    a = select(ROWS, "id", "target", ["light"], 3, seed=0)
    for label in ("0", "1"):
        assert {r["light"] for r in a if r["target"] == label} == {"day", "night", "dusk"}


def test_small_groups_are_not_padded():
    assert len(select(ROWS[:3], "id", "target", [], 10, seed=0)) == 3

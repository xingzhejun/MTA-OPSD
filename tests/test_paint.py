import pytest

from src.opsd_format import compute_paint_anchor_overlap, extract_paint_anchors


def test_paint_anchor_overlap_uses_answer_formula_and_numbers():
    solution = r"We obtain $x = 12$, hence the final result is \boxed{12}."
    anchors = extract_paint_anchors(solution, "12")

    assert "12" in anchors
    assert "x=12" in anchors
    overlap, returned_anchors = compute_paint_anchor_overlap(
        solution, "12", r"My calculation gives x=12 and therefore \boxed{12}."
    )
    assert returned_anchors == anchors
    assert overlap == pytest.approx(1.0)


def test_paint_anchor_overlap_defaults_to_zero_without_anchors():
    overlap, anchors = compute_paint_anchor_overlap("purely verbal reasoning", "", "anything")
    assert overlap == 0.0
    assert anchors == []


def test_paint_anchor_overlap_is_recall_over_unique_anchors():
    solution = r"First use $a=2$, then use $b=3$, and conclude \boxed{6}."
    overlap, anchors = compute_paint_anchor_overlap(solution, "6", "We found a=2.")
    matched = [anchor for anchor in anchors if anchor in "wefounda=2."]
    assert overlap == pytest.approx(len(matched) / len(anchors))

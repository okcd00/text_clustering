import importlib.util
from pathlib import Path
from xml.etree import ElementTree as ET

spec = importlib.util.spec_from_file_location(
    "esc", Path(__file__).resolve().parents[1] / "scripts/benchmark_esc_crossday.py")
esc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(esc)


def test_date_conservative():
    def source(*values):
        return ET.fromstring("<x>" + "".join(
            f'<time DCT="TRUE" value="{v}" />' for v in values) + "</x>")
    assert esc.dct(source("2009-12-03T14:36Z", "2009-12-03")) == "2009-12-03"
    assert esc.dct(source("2009-12-3T14:36Z")) is None
    assert esc.dct(source("2009-02-30")) is None
    assert esc.dct(source("2009-12-03", "2009-12-04")) is None


def test_no_same_day_visibility_and_prefix_stability():
    rows = [("01", "a", {"x": 1}), ("01", "b", {"x": 1}), ("02", "c", {"x": 1})]
    for adaptive in (False, True):
        assert esc.replay(rows, .6, adaptive) == {"a": "a", "b": "b", "c": "a"}
        assert esc.replay(rows[:2], .6, adaptive) == {"a": "a", "b": "b"}


def test_only_adaptive_accumulates_into_existing_centroid():
    rows = [("01", "a", {"x": 1}), ("02", "b", {"x": .6, "y": .8}),
            ("03", "c", {"y": 1})]
    assert esc.replay(rows, .4, False)["c"] == "c"
    assert esc.replay(rows, .4, True)["c"] == "a"


def test_cross_day_only_and_event_links_are_not_topic_labels():
    rows = [("01", "a", {}), ("01", "b", {}), ("02", "c", {}), ("03", "d", {})]
    scores = esc.metrics(rows, {"a": 1, "b": 1, "c": 1, "d": 2},
                         {"a": {"event"}, "c": {"event"}},
                         {"a": "a", "b": "a", "c": "c", "d": "a"})
    assert scores["cross_day_topic_pairs"] == {"tp": 0, "fp": 2, "fn": 2,
                                               "precision": 0, "recall": 0, "f1": 0}
    assert scores["confirmed_cross_day_event_link_pairs"] == {"captured": 0, "total": 1}
    assert scores["largest_cluster_rows"] == 3

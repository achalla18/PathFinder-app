import pytest

from pathfinder.graph.classify import PEDESTRIAN_WAY, STREET_CENTERLINE, classify, sidewalk_sides


@pytest.mark.parametrize(
    "tags,expected",
    [
        ({}, ("unknown", "unknown")),
        ({"sidewalk": "both"}, ("yes", "yes")),
        ({"sidewalk": "left"}, ("yes", "no")),
        ({"sidewalk": "right"}, ("no", "yes")),
        ({"sidewalk": "no"}, ("no", "no")),
        ({"sidewalk": "separate"}, ("separate", "separate")),
        ({"sidewalk:both": "separate"}, ("separate", "separate")),
        ({"sidewalk:left": "separate", "sidewalk:right": "no"}, ("separate", "no")),
        ({"sidewalk": "both", "sidewalk:right": "no"}, ("yes", "no")),  # specific overrides general
        ({"sidewalk": "weird"}, ("unknown", "unknown")),
    ],
)
def test_sidewalk_sides(tags, expected):
    assert sidewalk_sides(tags) == expected


@pytest.mark.parametrize(
    "tags,kind,rep",
    [
        ({"highway": "footway", "footway": "sidewalk"}, "sidewalk", PEDESTRIAN_WAY),
        ({"highway": "footway", "footway": "crossing"}, "crossing", PEDESTRIAN_WAY),
        ({"highway": "footway"}, "footway", PEDESTRIAN_WAY),
        ({"highway": "steps"}, "steps", PEDESTRIAN_WAY),
        ({"highway": "steps", "conveying": "yes"}, "escalator", PEDESTRIAN_WAY),
        ({"highway": "cycleway"}, "shared_path", PEDESTRIAN_WAY),
        ({"highway": "residential", "sidewalk": "both"}, "street_with_sidewalk", STREET_CENTERLINE),
        ({"highway": "residential", "sidewalk": "no"}, "street_no_sidewalk", STREET_CENTERLINE),
        ({"highway": "residential"}, "street_unknown", STREET_CENTERLINE),
        ({"highway": "service", "service": "alley"}, "alley", STREET_CENTERLINE),
        ({"highway": "living_street"}, "pedestrian_street", STREET_CENTERLINE),
        ({"highway": "trunk", "foot": "yes", "sidewalk": "both"}, "street_with_sidewalk", STREET_CENTERLINE),
    ],
)
def test_walkable_kinds(tags, kind, rep):
    c = classify(tags)
    assert (c.kind, c.representation) == (kind, rep)


@pytest.mark.parametrize(
    "tags",
    [
        {"highway": "cycleway", "foot": "no"},
        {"highway": "footway", "access": "private"},
        {"highway": "motorway"},
        {"highway": "trunk"},
        {"highway": "primary", "sidewalk": "separate"},
        {"highway": "primary", "sidewalk:left": "separate", "sidewalk:right": "no"},
        {"highway": "service", "service": "drive-through"},
        {"highway": "proposed"},
        {"highway": "platform"},
    ],
)
def test_not_walkable(tags):
    assert classify(tags).kind is None


def test_foot_yes_overrides_private_access():
    assert classify({"highway": "footway", "access": "private", "foot": "yes"}).kind == "footway"


def test_one_side_separate_other_unknown_needs_geometric_check():
    c = classify({"highway": "residential", "sidewalk:left": "separate"})
    assert c.kind == "street_unknown" and c.sidewalk_left == "separate"


def test_an_elevator_drawn_as_a_room_outline_is_not_a_path():
    assert classify({"highway": "elevator"}).kind == "elevator"
    assert classify({"highway": "elevator", "indoor": "room"}).kind is None
    assert classify({"highway": "elevator", "area": "yes"}).kind is None

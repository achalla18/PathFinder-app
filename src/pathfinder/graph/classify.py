"""Walkability rules: which OSM ways become pedestrian edges, and as what kind.

This module is pure functions over tag dicts so every rule is unit-tested.

Edge kinds
----------
Dedicated pedestrian space (``representation = "pedestrian_way"``):
    sidewalk, crossing, traffic_island, footway, path, pedestrian_street, steps,
    escalator, corridor, shared_path, access_aisle, elevator
Street centerlines (``representation = "street_centerline"``) — used only where pedestrians
have no separately mapped space, so the UI can say "sidewalk not mapped separately":
    street_with_sidewalk   OSM says a sidewalk exists but it is not mapped as its own way
    street_no_sidewalk     OSM says there is no sidewalk (walking in the roadway)
    street_unknown         no sidewalk information and none detected nearby
    alley, service
"""

from __future__ import annotations

from dataclasses import dataclass

PEDESTRIAN_WAY = "pedestrian_way"
STREET_CENTERLINE = "street_centerline"

STREET_HIGHWAYS = {
    "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link",
    "residential", "unclassified", "road", "living_street",
}
LIMITED_ACCESS = {"motorway", "motorway_link", "trunk", "trunk_link"}
NEVER_WALKABLE = {
    "proposed", "construction", "abandoned", "disused", "raceway", "bus_guideway",
    "platform", "busway", "escape", "rest_area", "services", "razed",
}
FOOT_ALLOWED = {"yes", "designated", "permissive", "official"}
FOOT_DENIED = {"no", "private", "use_sidepath", "discouraged"}


@dataclass(frozen=True)
class Classification:
    kind: str | None  # None => not a walking edge
    representation: str | None
    reason: str
    sidewalk_left: str = "n/a"  # yes | no | separate | unknown | n/a
    sidewalk_right: str = "n/a"


def sidewalk_sides(tags: dict) -> tuple[str, str]:
    """Normalize the many OSM sidewalk tagging schemes to (left, right) in
    {yes, no, separate, unknown}."""

    def norm(v: str | None) -> str:
        if v is None:
            return "unknown"
        v = v.strip().lower()
        if v in ("yes", "both"):
            return "yes"
        if v in ("no", "none"):
            return "no"
        if v == "separate":
            return "separate"
        return "unknown"

    left = right = "unknown"
    sw = (tags.get("sidewalk") or "").strip().lower() or None
    if sw in ("both", "yes"):
        left = right = "yes"
    elif sw == "left":
        left, right = "yes", "no"
    elif sw == "right":
        left, right = "no", "yes"
    elif sw in ("no", "none"):
        left = right = "no"
    elif sw == "separate":
        left = right = "separate"
    if tags.get("sidewalk:both"):
        left = right = norm(tags["sidewalk:both"])
    if tags.get("sidewalk:left"):
        left = norm(tags["sidewalk:left"])
    if tags.get("sidewalk:right"):
        right = norm(tags["sidewalk:right"])
    return left, right


def _foot_denied(tags: dict) -> bool:
    foot = tags.get("foot")
    if foot in FOOT_ALLOWED:
        return False
    if foot in FOOT_DENIED:
        return True
    return tags.get("access") in ("no", "private")


def classify(tags: dict) -> Classification:
    hw = tags.get("highway")
    if hw is None or hw in NEVER_WALKABLE:
        return Classification(None, None, f"highway={hw} not walkable")
    if _foot_denied(tags):
        return Classification(None, None, "foot/access denied")

    # ---- dedicated pedestrian space
    if hw == "footway":
        fw = tags.get("footway")
        kind = {
            "sidewalk": "sidewalk",
            "crossing": "crossing",
            "traffic_island": "traffic_island",
            "access_aisle": "access_aisle",
        }.get(fw, "footway")
        return Classification(kind, PEDESTRIAN_WAY, f"footway={fw}")
    if hw == "steps":
        kind = "escalator" if tags.get("conveying") else "steps"
        return Classification(kind, PEDESTRIAN_WAY, "steps")
    if hw == "elevator":
        # An elevator drawn as the outline of its shaft or car (indoor=room, or an area) is a
        # room, not a path: walking its outline made "links" up to 43 m long at 9th & G St NW.
        if tags.get("indoor") in ("room", "area") or tags.get("area") == "yes":
            return Classification(None, None, "elevator mapped as a room outline, not a path")
        return Classification("elevator", PEDESTRIAN_WAY, "elevator way")
    if hw == "pedestrian":
        return Classification("pedestrian_street", PEDESTRIAN_WAY, "pedestrian")
    if hw == "corridor":
        return Classification("corridor", PEDESTRIAN_WAY, "corridor")
    if hw in ("path", "track", "bridleway"):
        return Classification("path", PEDESTRIAN_WAY, hw)
    if hw == "cycleway":
        # US OSM access defaults allow pedestrians on cycleways unless foot=no; many DC
        # trails are mapped this way. Marked as shared so the cost model can tell it apart.
        return Classification("shared_path", PEDESTRIAN_WAY, "cycleway (foot not denied)")

    # ---- limited-access roads: only with explicit pedestrian provision
    if hw in LIMITED_ACCESS:
        left, right = sidewalk_sides(tags)
        if tags.get("foot") in FOOT_ALLOWED and "separate" not in (left, right):
            return Classification("street_with_sidewalk" if "yes" in (left, right) else "street_unknown",
                                  STREET_CENTERLINE, f"{hw} with foot={tags.get('foot')}", left, right)
        return Classification(None, None, f"{hw} without pedestrian provision")

    # ---- service roads
    if hw == "service":
        svc = tags.get("service")
        if svc == "drive-through":
            return Classification(None, None, "drive-through")
        left, right = sidewalk_sides(tags)
        if left == "separate" and right == "separate":
            return Classification(None, None, "service road with separate sidewalks")
        kind = "alley" if svc == "alley" else "service"
        return Classification(kind, STREET_CENTERLINE, f"service={svc}", left, right)

    # ---- streets
    if hw in STREET_HIGHWAYS:
        left, right = sidewalk_sides(tags)
        if hw == "living_street":
            return Classification("pedestrian_street", STREET_CENTERLINE, "living_street", left, right)
        sides = {left, right}
        if sides <= {"separate", "no"} and "separate" in sides:
            return Classification(None, None, "sidewalks mapped separately", left, right)
        if "yes" in sides:
            return Classification("street_with_sidewalk", STREET_CENTERLINE, "sidewalk tagged on street", left, right)
        if sides == {"no"}:
            return Classification("street_no_sidewalk", STREET_CENTERLINE, "sidewalk=no", left, right)
        # unknown on at least one side (possibly separate on the other): geometric detection
        # in graph.build decides whether separate sidewalks exist nearby.
        return Classification("street_unknown", STREET_CENTERLINE, "no sidewalk tags", left, right)

    return Classification(None, None, f"highway={hw} unhandled")

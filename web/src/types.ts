// Shapes of the planner's answer (see src/pathfinder/routing/service.py).

export type LonLat = [number, number];

export interface Claim {
  value?: string | number | boolean | null;
  status: "verified" | "inferred" | "unknown";
  source: string | null;
  as_of: string | null;
}

export interface Condition extends Partial<Claim> {
  type: string;
  street?: string | null;
  control?: Claim;
  curb_ramps?: Claim[] | Claim;
  ramp_checks?: Claim[];
  length_m?: number;
  count?: number;
  note?: string;
}

/** Felt minutes per accessibility dimension (routing/cost.py). */
export interface Felt { cross: number; slope: number; sidewalk: number }

export interface Side {
  length_m: number;
  geometry: LonLat[];
  conditions: Condition[];
}

export interface Place {
  at: LonLat;
  extra_minutes: number;
  felt_minutes_saved: Felt;
  net_felt_minutes_saved: number;
  this_route: Side;
  fastest_route: Side;
}

export interface Comparison {
  extra_minutes: number;
  felt_minutes_saved: number;
  places_that_differ: number;
  crux: Place | null;
  other_places: Place[];
}

export interface RouteSummary {
  minutes: number;
  length_m: number;
  felt_minutes: Felt;
  felt_total_minutes: number;
  geometry: LonLat[];
}

/** One thing to do or to know about along a route, in walking order (routing/guidance.py). */
export interface Step extends Partial<Claim> {
  type: "start" | "turn" | "cross" | "unmapped_crossing" | "steps" | "no_sidewalk" | "arrive";
  vertex: number; // index into the route's geometry: where the step is
  heading?: string; // start: the compass direction to set off in
  direction?: "left" | "right" | "slight_left" | "slight_right" | "sharp_left" | "sharp_right";
  onto?: string; // turn: the way turned onto, when it has a name
  street?: string | null; // cross: the street crossed
  length_m?: number;
  escalator?: boolean;
  control?: Claim;
  curb_ramps?: Claim[] | Claim;
}

export interface RouteOption extends RouteSummary {
  id: number;
  steps?: Step[];
  is_fastest: boolean;
  is_recommended: boolean;
  priority_region_share: number;
  compared_with_fastest?: Comparison;
}

export interface SnapInfo {
  snap_distance_m: number;
  walkway_meets_requirements: boolean;
  nearer_walkway_passed_over: boolean;
  inside_closed_area?: string; // the point is inside an area closed to the public (OSM landuse=military)
}

export interface Relaxation {
  compromise: string;
  route: RouteSummary;
  where_it_applies: Condition[];
  edges_outside_original_requirements: number;
}

/** A Metro elevator out of service (or of unknown status) at a station the trip starts or ends at. */
export interface TransitNotice {
  end: "origin" | "destination";
  station: string;
  status: "out_of_service" | "unknown";
  where?: string | null;
  reason?: string | null;
  estimated_return_eastern?: string | null;
  message?: string;
  source: string;
  as_of?: string | null;
}

export interface Plan {
  transit_notices?: TransitNotice[];
  status: "ok" | "no_feasible_route" | "not_near_network";
  disclaimer: string;
  message?: string;
  note?: string;
  origin?: SnapInfo;
  destination?: SnapInfo;
  route_options?: RouteOption[];
  triangle?: [number, number, number, number][]; // crossings, slope, sidewalk, winning option id
  relaxation?: Relaxation | null;
}

export type Certainty = "verified_only" | "allow_likely" | "allow_unknown";

export interface Settings {
  profile: string; // the mobility profile the settings started from ("custom" once changed by hand)
  stepFree: boolean;
  maxGrade: number | null;
  certainty: Certainty;
  cross: number;
  slope: number;
  sidewalk: number;
  budgetMin: number;
}

/** A mobility profile: starting requirements and priorities (routing/profiles.py). */
export interface Profile {
  id: string;
  label: string;
  description: string;
  requirements: { step_free: boolean; max_grade_pct: number | null; certainty: Certainty };
  mix: { cross: number; slope: number; sidewalk: number };
}

/** One recorded barrier or unrecorded place on the city barrier map (analysis/barriers.py). */
export interface BarrierProps {
  type: string;
  kind: "barrier" | "unknown" | "evidence"; // evidence: an estimated sidewalk line
  what: string;
  status: Claim["status"];
  source: string | null;
  as_of: string | null;
  length_m: number | null;
  street: string | null;
}

export interface Barriers {
  type: "FeatureCollection";
  features: { type: "Feature"; properties: BarrierProps; geometry: { type: string; coordinates: unknown } }[];
  truncated: boolean;
  counts: Record<string, number>; // every record in the view, also when the features are capped
  // sidewalk lines estimated from a survey where OSM maps only the street (not barriers)
  sidewalks?: { features: Barriers["features"]; count: number; truncated: boolean };
  unlicensed_sources: string[];
}

export interface PlaceResult {
  label: string;
  detail: string;
  kind: string;
  lon: number;
  lat: number;
}

// The barrier map's colours and legend, shared by the map and the panel (kept out of the
// map module so the panel does not pull in the map library).

// Barriers in strong colours, unknowns in grey: for a city the grey is the survey worklist.
export const BARRIER_COLORS: Record<string, string> = {
  curb_barrier: "#b30000",
  junction_curb_barrier: "#b30000",
  ramp_fails_check: "#f08c00",
  ramp_no_tactile_pad: "#1c7ed6",
  steep: "#8e4ec6",
  steps: "#d9480f",
  no_sidewalk: "#e4572e",
  sidewalk_poor_condition: "#8c5a1e",
  sidewalk_narrow: "#0b7285",
  curb_unknown: "#8a94a3",
  junction_curb_unknown: "#8a94a3",
  sidewalk_unknown: "#a3acb8",
};

// Sidewalk lines estimated from a survey (evidence, drawn dashed under the barriers).
export const ESTIMATED_SIDEWALK_COLOR = "#2f9e44";

// Legend rows: one per thing a person would look for, recorded barriers first.
export const LEGEND: { types: string[]; label: string; kind: "barrier" | "unknown" }[] = [
  { types: ["curb_barrier", "junction_curb_barrier"], label: "Raised curb or no curb ramp", kind: "barrier" },
  { types: ["ramp_fails_check"], label: "Ramp fails the city's check", kind: "barrier" },
  { types: ["ramp_no_tactile_pad"], label: "Ramp without a tactile pad", kind: "barrier" },
  { types: ["no_sidewalk"], label: "No sidewalk: walking in the roadway", kind: "barrier" },
  { types: ["sidewalk_poor_condition"], label: "Sidewalk condition under 40 of 100 (city survey)", kind: "barrier" },
  { types: ["sidewalk_narrow"], label: "Sidewalk narrower than 4 ft", kind: "barrier" },
  { types: ["steep"], label: "Steeper than 8.33% (ADA ramp limit)", kind: "barrier" },
  { types: ["steps"], label: "Steps", kind: "barrier" },
  { types: ["curb_unknown", "junction_curb_unknown"], label: "Curb not recorded", kind: "unknown" },
  { types: ["sidewalk_unknown"], label: "Sidewalk not recorded", kind: "unknown" },
];

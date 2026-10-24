// What the chosen route avoids, shown on the map: at each place where it leaves the fastest
// route, the worst thing on the fastest route's stretch that the chosen stretch does not have.
// The label sits halfway along the fastest route's stretch: it marks the section, not a point
// (the planner's conditions carry no coordinates of their own).

import { cumulative, pointAt } from "./geo";
import { t } from "./i18n";
import type { Condition, LonLat, Place, Side } from "./types";

export interface Hazard { at: LonLat; kind: Kind; label: string }
type Kind = "steps" | "no_ramp" | "steep" | "no_sidewalk" | "curb_unknown" | "unmarked" | "no_signal";

const STEEP_PCT = 8.33; // the ADA ramp limit: steeper than a wheelchair ramp may be
// most serious first: a kind that both stretches share equally is not what the route avoids
const ORDER: Kind[] = ["steps", "no_ramp", "steep", "no_sidewalk", "curb_unknown", "unmarked", "no_signal"];

function curbs(c: Condition) {
  return !c.curb_ramps ? [] : Array.isArray(c.curb_ramps) ? c.curb_ramps : [c.curb_ramps];
}

/** How much of each kind of problem a stretch has, and the steepest grade on it. */
function tally(side: Side): { n: Record<Kind, number>; steepest: number } {
  const n: Record<Kind, number> = { steps: 0, no_ramp: 0, steep: 0, no_sidewalk: 0, curb_unknown: 0, unmarked: 0, no_signal: 0 };
  let steepest = 0;
  for (const c of side.conditions) {
    if (c.type === "steps") n.steps += 1;
    else if (c.type === "no_sidewalk") n.no_sidewalk += c.length_m ?? 1;
    else if (c.type === "steepest_sustained_grade_pct") {
      const g = Number(c.value);
      if (g > STEEP_PCT) { n.steep += 1; steepest = Math.max(steepest, g); }
    } else if (c.type === "crossing" || c.type === "unmapped_crossing") {
      for (const k of curbs(c)) {
        if (k.value === "no_ramp" || k.value === "raised" || k.value === "no_ramp_reported") n.no_ramp += 1;
        else if (!k.value || k.status === "unknown") n.curb_unknown += 1;
      }
      if (c.control?.value === "unmarked" || c.type === "unmapped_crossing") n.unmarked += 1;
      else if (c.control?.value === "marked_uncontrolled") n.no_signal += 1;
    }
  }
  return { n, steepest };
}

function label(kind: Kind, steepest: number): string {
  switch (kind) {
    case "steps": return t("Steps");
    case "no_ramp": return t("No curb ramp");
    case "steep": return t("Steep {g}%", { g: steepest.toFixed(1) });
    case "no_sidewalk": return t("No sidewalk");
    case "curb_unknown": return t("Curb not recorded");
    case "unmarked": return t("Unmarked crossing");
    case "no_signal": return t("Crossing with no signal");
  }
}

/** One label per place, where the fastest route has something the chosen route avoids. */
export function hazards(places: Place[]): Hazard[] {
  const out: Hazard[] = [];
  for (const pl of places) {
    const line = pl.fastest_route.geometry;
    if (line.length < 2) continue;
    const fast = tally(pl.fastest_route), mine = tally(pl.this_route);
    const kind = ORDER.find((k) => fast.n[k] > mine.n[k]);
    if (!kind) continue;
    const cum = cumulative(line);
    out.push({ at: pointAt(line, cum, cum[cum.length - 1] / 2), kind, label: label(kind, fast.steepest) });
  }
  return out;
}

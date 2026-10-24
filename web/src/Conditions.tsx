import type { Claim, Condition } from "./types";
import { plural, t } from "./i18n";

// Every statement shown to a user carries its status, source and date (product rule 4).

const SOURCES: Record<string, string> = {
  osm_dc: "OpenStreetMap",
  montgomery_sidewalks: "Montgomery Planning sidewalk survey",
  frederick_sidewalks: "City of Frederick sidewalk survey",
  frederick_ramps: "City of Frederick ramp survey",
  gaithersburg_dtm: "Maryland LiDAR terrain",
  frederick_dtm: "Maryland LiDAR terrain",
  montgomery_signals: "Montgomery County signal inventory",
  frederick_county_signals: "Frederick County signal inventory",

  sidewalks_2021: "DC planimetric survey",
  sidewalk_ramps_2010: "DC ramp survey",
  project_sidewalk_dc: "Project Sidewalk",
  traffic_signals: "DDOT signal inventory",
  dtm_2024: "DC LiDAR terrain",
  roadway_block: "DDOT roadway inventory",
};

export const RAMP_CHECK: Record<string, string> = {
  passed: "ramp passed the city's check",
  fails: "ramp fails the city's check (too steep, no landing or damaged)",
  no_tactile_pad: "ramp has no working tactile warning pad",
};

export const CONTROL: Record<string, string> = {
  signal: "signalized",
  marked_uncontrolled: "painted, no signal",
  unmarked: "unmarked, no signal",
};

export const KERB: Record<string, string> = {
  lowered: "lowered curb",
  flush: "flush curb",
  ramp_present: "curb ramp",
  raised: "raised curb",
  no_ramp: "no curb ramp",
  no_curb: "no curb to mount",
  ramp_likely: "ramps recorded at the corners",
  no_ramp_reported: "a missing ramp is reported here",
  unknown: "curb not recorded",
};

/** A street as the roadway inventory names it ("MINNESOTA AVE NE"), in ordinary case with the quadrant kept. */
export const streetName = (s: string) =>
  s.toLowerCase().replace(/(^|[\s-])([a-z])/g, (_, gap: string, c: string) => gap + c.toUpperCase()).replace(/\b(Nw|Ne|Sw|Se)$/, (q) => q.toUpperCase());

function backing(c: Partial<Claim>): string {
  if (!c.status || c.status === "unknown") return t("no record");
  // a claim merged from two halves of a crossing may name several sources, ", "-separated
  const label = (s: string) => t(SOURCES[s] ?? (s.startsWith("osm_") ? "OpenStreetMap" : s.startsWith("derived") ? "estimated" : s));
  const src = c.source ? [...new Set(c.source.split(", ").map(label))].join(` ${t("and")} `) : "";
  const year = c.as_of ? ` ${String(c.as_of).slice(0, 4)}` : "";
  return `${t(c.status === "inferred" ? "likely" : "recorded")}${src ? ` · ${src}` : ""}${year}`;
}

export function Fact({ text, claim }: { text: string; claim: Partial<Claim> }) {
  return (
    <li className={`fact ${claim.status ?? "unknown"}`}>
      <span>{t(text)}</span>
      <small>{backing(claim)}</small>
    </li>
  );
}

export function kerbs(c: { curb_ramps?: Claim[] | Claim }): Claim[] {
  if (!c.curb_ramps) return [];
  return Array.isArray(c.curb_ramps) ? c.curb_ramps : [c.curb_ramps];
}

export function Conditions({ items }: { items: Condition[] }) {
  if (items.length === 0) return <p className="muted">{t("Nothing notable is recorded on this stretch.")}</p>;
  return (
    <ul className="facts">
      {items.map((c, i) => {
        const key = `${c.type}-${i}`;
        if (c.type === "crossing" || c.type === "unmapped_crossing") {
          const control = c.control?.value ? (CONTROL[String(c.control.value)] ?? String(c.control.value)) : "control not recorded";
          const name = c.type !== "crossing" ? t("Street crossing that is not mapped")
            : c.street ? t("Crossing of {street}", { street: streetName(c.street) }) : t("Crossing");
          return (
            <li key={key} className="group">
              <strong>{name}</strong>
              <ul>
                <Fact text={control} claim={c.control ?? { status: "unknown" }} />
                {kerbs(c).map((k, j) => (
                  <Fact key={j} text={k.value ? (KERB[String(k.value)] ?? String(k.value)) : "curb not recorded"} claim={k} />
                ))}
                {(c.ramp_checks ?? []).filter((k) => k.status !== "unknown" && k.value).map((k, j) => (
                  <Fact key={`check-${j}`} text={RAMP_CHECK[String(k.value)] ?? String(k.value)} claim={k} />
                ))}
              </ul>
            </li>
          );
        }
        const m = c.length_m != null ? `${Math.round(c.length_m)} m` : "";
        const text: Record<string, string> = {
          steps: t("Steps ({m})", { m }),
          steepest_sustained_grade_pct: t("Steepest sustained slope {g}%", { g: String(c.value) }),
          slope_unknown: t("Slope not measured for {m}", { m }),
          no_sidewalk: t("No sidewalk for {m}", { m }),
          sidewalk_one_side_recorded: t("Sidewalk recorded on one side for {m}; the other side is not recorded", { m }),
          lowest_sidewalk_condition: t("Lowest sidewalk condition on record: {v} of 100 (the survey's index, higher is better)", { v: Math.round(Number(c.value)) }),
          narrowest_sidewalk_width_m: t("Narrowest sidewalk on record: {v} m", { v: String(c.value) }),
          inferred_link: plural(c.count ?? 0, "{n} short link assumed walkable (gap in the map)", "{n} short links assumed walkable (gap in the map)"),
        };
        return <Fact key={key} text={text[c.type] ?? c.type} claim={c} />;
      })}
    </ul>
  );
}

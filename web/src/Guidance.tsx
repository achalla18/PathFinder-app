import { CONTROL, Fact, KERB, kerbs, streetName } from "./Conditions";
import { metres } from "./geo";
import { t } from "./i18n";
import type { Step } from "./types";

// On-screen guidance while walking: the next thing to do, how far ahead it is, and what is
// recorded about it. A crossing's control and curbs are shown as claims with their source
// and date, like every other statement on the page (product rule 4).

const PASSED_M = 8; // a step this far behind the walker is done with
const NOW_M = 15; // closer than this, the step is "now" rather than a distance away

export interface Upcoming {
  step: Step;
  ahead: number; // metres from the walker to the step; negative while they are on it
}

/** The steps not yet passed, nearest first. `cum` is cumulative distance along the route's geometry. */
export function upcoming(steps: Step[], cum: number[], along: number): Upcoming[] {
  return steps
    .filter((s) => s.type !== "start")
    .map((step) => ({ step, ahead: (cum[step.vertex] ?? cum[cum.length - 1]) - along }))
    // a crossing or a flight of stairs is still the step until the walker is off its far end
    .filter(({ step, ahead }) => step.type === "arrive" || ahead + (step.type === "turn" ? 0 : step.length_m ?? 0) > -PASSED_M);
}

const TURNS: Record<NonNullable<Step["direction"]>, [string, number]> = {
  left: ["Turn left", -90],
  right: ["Turn right", 90],
  slight_left: ["Bear left", -45],
  slight_right: ["Bear right", 45],
  sharp_left: ["Sharp left", -135],
  sharp_right: ["Sharp right", 135],
};

// A way's own name comes from OpenStreetMap in ordinary case; a street beside a sidewalk
// comes from the roadway inventory in capitals.
const wayName = (s: string) => (s === s.toUpperCase() ? streetName(s) : s);

export function stepTitle(s: Step): string {
  switch (s.type) {
    case "turn": {
      const turn = t(TURNS[s.direction ?? "left"][0]);
      return s.onto ? t("{turn} onto {street}", { turn, street: wayName(s.onto) }) : turn;
    }
    case "cross": return s.street ? t("Cross {street}", { street: streetName(s.street) }) : t("Cross the street");
    case "unmapped_crossing": return t("Cross at the junction");
    case "steps": return t(s.escalator ? "Escalator" : "Stairs");
    case "no_sidewalk": return t("No sidewalk");
    case "arrive": return t("Your destination");
    default: return "";
  }
}

function Glyph({ step }: { step: Step }) {
  const stroke = { fill: "none", stroke: "currentColor", strokeWidth: 3, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  if (step.type === "turn") {
    return <svg viewBox="0 0 30 30" aria-hidden="true"><g transform={`rotate(${TURNS[step.direction ?? "left"][1]} 15 15)`} {...stroke}><path d="M15 25V6M8 13l7-7 7 7" /></g></svg>;
  }
  if (step.type === "cross" || step.type === "unmapped_crossing") {
    return <svg viewBox="0 0 30 30" aria-hidden="true"><g {...stroke} strokeWidth={4}><path d="M6 6v18M15 6v18M24 6v18" /></g></svg>;
  }
  if (step.type === "steps") return <svg viewBox="0 0 30 30" aria-hidden="true"><path d="M4 25h7v-7h7v-7h8" {...stroke} /></svg>;
  if (step.type === "arrive") return <svg viewBox="0 0 30 30" aria-hidden="true"><circle cx="15" cy="15" r="8" {...stroke} /><circle cx="15" cy="15" r="2.5" fill="currentColor" /></svg>;
  return <svg viewBox="0 0 30 30" aria-hidden="true"><path d="M15 6v12M15 24v.5" {...stroke} strokeWidth={4} /></svg>;
}

/** What is on record about a step, each with its backing. Unknown is said, not left out. */
function Facts({ step }: { step: Step }) {
  if (step.type === "cross" || step.type === "unmapped_crossing") {
    const control = step.control?.value ? (CONTROL[String(step.control.value)] ?? String(step.control.value)) : "control not recorded";
    return (
      <ul className="facts">
        {step.type === "unmapped_crossing" && <Fact text="this crossing is not mapped" claim={{ status: "unknown" }} />}
        <Fact text={control} claim={step.control ?? { status: "unknown" }} />
        {kerbs(step).map((k, j) => <Fact key={j} text={k.value ? (KERB[String(k.value)] ?? String(k.value)) : "curb not recorded"} claim={k} />)}
      </ul>
    );
  }
  if (step.type === "steps" || step.type === "no_sidewalk") {
    const m = step.length_m != null ? `${Math.round(step.length_m)} m` : "";
    return <ul className="facts"><Fact text={step.type === "steps" ? `${t(step.escalator ? "escalator" : "steps")}, ${m}` : t("walk in the roadway for {m}", { m })} claim={step} /></ul>;
  }
  return null;
}

const lower = (s: string) => s.replace(/^./, (c) => c.toLowerCase());

export function NextStep({ next, then }: { next: Upcoming; then: Upcoming | null }) {
  return (
    <div className="step" aria-live="polite">
      <span className="step-glyph"><Glyph step={next.step} /></span>
      <h2>{stepTitle(next.step)}<small>{next.ahead > NOW_M ? t("in {d}", { d: metres(next.ahead) }) : t("now")}</small></h2>
      <Facts step={next.step} />
      {then && <p className="then">{then.ahead - next.ahead > NOW_M
        ? t("Then {step} after {d}", { step: lower(stepTitle(then.step)), d: metres(then.ahead - next.ahead) })
        : t("Then {step}", { step: lower(stepTitle(then.step)) })}</p>}
    </div>
  );
}

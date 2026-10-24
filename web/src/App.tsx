import { Component, lazy, type ReactNode, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { barriersIn, basemapInfo, cachedProfiles, planTrip, profiles as fetchProfiles, type ReportKind, sendReport, townInfo, type Example, type PlaceLink } from "./api";
import { BARRIER_COLORS, ESTIMATED_SIDEWALK_COLOR, LEGEND } from "./barrierStyle";
import { Conditions, Fact, streetName } from "./Conditions";
import { Feedback } from "./Feedback";
import { hazards as hazardsOf } from "./hazards";
import { plural, setLang, t, tProfile, useLang } from "./i18n";
import { cumulative, distance, metres, pointAt, project, split } from "./geo";
import { NextStep, upcoming } from "./Guidance";
import { SearchBox } from "./SearchBox";
import { gridKey, type Mix, snap, Triangle } from "./Triangle";
import type { BarrierProps, Barriers, Certainty, LonLat, Place, Plan, Profile, RouteOption, Settings } from "./types";

const DEFAULTS: Settings = { profile: "custom", stepFree: false, maxGrade: null, certainty: "allow_likely", cross: 1 / 3, slope: 1 / 3, sidewalk: 1 / 3, budgetMin: 5 };
const CERTAINTY: [Certainty, string][] = [["verified_only", "Recorded only"], ["allow_likely", "Recorded or likely"], ["allow_unknown", "Anything not ruled out"]];
const SLOPES: [number | null, string][] = [[null, "Any slope"], [5, "≤ 5%"], [8.33, "≤ 8.3%"], [10, "≤ 10%"]];
// One colour per route, shared by its line on the map, its card and its territory in the
// triangle. The fastest route is always the neutral one.
const FASTEST_COLOR = "#64748b";
const ROUTE_COLORS = ["#2f6df6", "#12a594", "#e4572e", "#8e4ec6", "#b83289", "#c88a04", "#30a46c", "#0e7490"];
const DIMS = [["cross", "Crossings & curbs", "var(--cross)"], ["slope", "Slope", "var(--slope)"], ["sidewalk", "Sidewalk", "var(--sidewalk)"]] as const;
// Walking guidance
const OFF_ROUTE_M = 40; // farther than this from the route, twice in a row, and the trip is re-planned
const ARRIVED_M = 20; // route left to walk...
const ARRIVED_RADIUS_M = 30; // ...and straight-line distance to its end, both needed to have arrived
const NEXT_PLACE_ALERT_M = 80;
const THEN_WITHIN_M = 30; // a second step this close behind the next one is announced with it
const START_HINT_M = 25; // which way to set off is said until the walker is this far along
// ?demo in the address replaces the phone's position with one that walks the chosen route,
// so guidance can be shown on a desk or in a presentation: the position stands at the trip's
// start until walking begins, then covers the whole route in about DEMO_ROUTE_S seconds.
const DEMO = new URLSearchParams(window.location.search).has("demo");
const DEMO_ROUTE_S = 18;
const DEMO_TICK_MS = 100; // ten positions a second; the map glides between them

// The map library is most of the page's weight. It is fetched and started only once the search
// card and the panel have been painted: starting the map's graphics first held the whole page
// blank on a phone. (Preloading its code alongside the page was measured too: on a slow
// connection it took bandwidth from the page itself and made everything later.)
// A failed download leaves a message and a retry over the map, never a blank page.
const loadMap = () => lazy(() => import("./MapView").then((m) => ({ default: m.MapView })));
void basemapInfo(); // asked now, so the answer is waiting when the map starts

class MapBoundary extends Component<{ children: ReactNode; onRetry: () => void }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="map map-failed">
        <p>{t("The map could not load. Search and routes still work.")}</p>
        <button className="btn ghost" onClick={() => { this.setState({ failed: false }); this.props.onRetry(); }}>{t("Try again")}</button>
      </div>
    );
  }
}

/** After the first paint: two animation frames, so the browser has drawn the page once. */
function useAfterFirstPaint(): boolean {
  const [done, setDone] = useState(false);
  useEffect(() => {
    let id = requestAnimationFrame(() => { id = requestAnimationFrame(() => setDone(true)); });
    return () => cancelAnimationFrame(id);
  }, []);
  return done;
}

/** Whether the phone says it is online, followed as it changes. */
function useOnline(): boolean {
  const [online, setOnline] = useState(() => navigator.onLine !== false);
  useEffect(() => {
    const on = () => setOnline(navigator.onLine !== false);
    window.addEventListener("online", on); window.addEventListener("offline", on);
    return () => { window.removeEventListener("online", on); window.removeEventListener("offline", on); };
  }, []);
  return online;
}

// The panel on a phone: pulled down to its button, at its usual height, or up over most of the map.
type Sheet = "low" | "mid" | "full";
const SHEET_ORDER: Sheet[] = ["low", "mid", "full"];
const DRAG_START_PX = 8; // a finger moving less than this is a tap, not a drag
const DRAG_SNAP_PX = 48; // and dragged this far, the panel moves to the next height

// What matters to someone rarely changes between walks, so it is remembered on this device
// (never sent anywhere), along with the last few places picked.
const STORE = "pathfinder.v2"; // v2: accessibility dimensions and profiles
type End = { at: LonLat; label: string };
interface Saved { settings?: Settings; recent?: End[]; recordWalks?: boolean }

// A walk recorded for tuning guidance: where the phone said it was, how far that was from
// the route, and what the page did about it. It is kept in memory on this device and goes
// nowhere unless the walker saves the file and passes it on themselves.
interface Track {
  started: number;
  line: LonLat[] | null;
  samples: { t: number; lat: number; lon: number; accuracy_m: number | null; off_route_m: number; along_m: number; next_place_m: number | null; next_step_m: number | null }[];
  events: { t: number; type: "route" | "replanned" | "arrived"; off_route_m?: number; geometry?: LonLat[] }[];
}

function load(): Saved {
  try { return JSON.parse(localStorage.getItem(STORE) ?? "{}") as Saved; } catch { return {}; }
}
// Writes are gathered and made at most every half second: a drag in the triangle changes the
// settings on every movement of the finger, and each write re-read and re-wrote the whole store.
let pending: Saved = {};
let saveTimer: ReturnType<typeof setTimeout> | null = null;
function flush() {
  saveTimer = null;
  try { localStorage.setItem(STORE, JSON.stringify({ ...load(), ...pending })); } catch { /* private browsing: nothing is remembered */ }
  pending = {};
}
function save(change: Saved) {
  pending = { ...pending, ...change };
  saveTimer ??= setTimeout(flush, 500);
}
window.addEventListener("pagehide", () => { if (saveTimer) { clearTimeout(saveTimer); flush(); } });

// The trip lives in the address after the "#", which browsers do not send to a server, so
// a reload or a shared link restores it.
function tripFromHash(): { from?: End; to?: End } {
  const q = new URLSearchParams(window.location.hash.slice(1));
  const end = (k: string): End | undefined => {
    const [lat, lon, ...name] = (q.get(k) ?? "").split(",");
    return lat && lon && isFinite(+lat) && isFinite(+lon) ? { at: [+lon, +lat], label: name.join(",") || "Pinned on the map" } : undefined;
  };
  return { from: end("from"), to: end("to") };
}
function tripToHash(from: End | null, to: End | null) {
  const q = new URLSearchParams();
  for (const [k, e] of [["from", from], ["to", to]] as const) if (e) q.set(k, `${e.at[1].toFixed(5)},${e.at[0].toFixed(5)},${e.label}`);
  history.replaceState(null, "", `${window.location.pathname}${window.location.search}${q.toString() ? `#${q}` : ""}`);
}

const min = (m: number) => `${m.toFixed(m < 10 ? 1 : 0)} min`;
const clock = (minutesFromNow: number) => new Date(Date.now() + minutesFromNow * 60000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

function places(o: RouteOption | undefined): Place[] {
  const c = o?.compared_with_fastest;
  return c ? [...(c.crux ? [c.crux] : []), ...c.other_places] : [];
}

/** The phone's own light or dark setting, followed as it changes. */
function useDarkTheme(): boolean {
  const q = typeof window.matchMedia === "function" ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  const [dark, setDark] = useState(Boolean(q?.matches));
  useEffect(() => {
    if (!q) return;
    const on = () => setDark(q.matches);
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return dark;
}

export function App() {
  const dark = useDarkTheme();
  const online = useOnline();
  const language = useLang(); // re-renders the page when the language changes
  // a barrier report being made from the Barriers tab: placed by a tap on the map
  const [report, setReport] = useState<ReportDraft | null>(null);
  const [townName, setTownName] = useState("");
  const [home, setHome] = useState<string | null>(null);
  const [placeLinks, setPlaceLinks] = useState<PlaceLink[]>([]);
  const [examples, setExamples] = useState<Example[]>([]);
  // with example trips to offer and nothing chosen yet, open on the tab that shows them
  useEffect(() => { void townInfo().then((t) => {
    setTownName(t.name); setHome(t.home); setPlaceLinks(t.places); setExamples(t.examples);
    if (t.examples.length && !tripFromHash().to) setTab((cur) => (cur === "priorities" ? "routes" : cur));
  }); }, []);
  const painted = useAfterFirstPaint();
  const [MapView, setMapView] = useState(loadMap);
  const [profileList, setProfileList] = useState<Profile[]>(cachedProfiles);
  useEffect(() => { void fetchProfiles().then(setProfileList); }, []);
  const [settings, setSettingsState] = useState<Settings>(() => ({ ...DEFAULTS, ...load().settings }));
  // The newest settings, readable at once: a drag can end (and ask for routes) before React
  // has re-rendered with the last position, and a request must never go out with older ones.
  const latest = useRef(settings);
  const setSettings = (change: (s: Settings) => Settings) => {
    latest.current = change(latest.current);
    setSettingsState(latest.current);
    save({ settings: latest.current });
  };
  const [gps, setGps] = useState<LonLat | null>(null);
  const [locationNote, setLocationNote] = useState<string | null>(null);
  const [pinnedOrigin, setPinnedOrigin] = useState<End | null>(() => tripFromHash().from ?? null);
  const [destination, setDestination] = useState<End | null>(() => tripFromHash().to ?? null);
  const [recent, setRecent] = useState<End[]>(() => load().recent ?? []);
  const [tapSets, setTapSets] = useState<"destination" | "origin">("destination");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [openPlace, setOpenPlace] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"priorities" | "routes" | "barriers">("priorities");
  const [view, setView] = useState<[number, number, number, number] | null>(null);
  const [barriers, setBarriers] = useState<Barriers | null>(null);
  const [barrierError, setBarrierError] = useState<string | null>(null);
  const [barrier, setBarrier] = useState<BarrierProps | null>(null);
  const [walking, setWalking] = useState(false);
  const [demoAt, setDemoAt] = useState<LonLat | null>(null);
  const [replanned, setReplanned] = useState(0);
  const [mixMoved, setMixMoved] = useState(false);
  const [recording, setRecording] = useState(() => Boolean(load().recordWalks));
  const [recorded, setRecorded] = useState(0); // samples in the walk recording, for the save button
  const [feedback, setFeedback] = useState<"closed" | "open" | "after-walk">("closed");
  const [sheet, setSheet] = useState<Sheet>("mid"); // on a phone: how much of the screen the panel takes
  const [recenter, setRecenter] = useState(0); // bumped by the locate button
  const [following, setFollowing] = useState(true); // while walking: the map is keeping up with the user
  const sheetEl = useRef<HTMLElement>(null);
  const drag = useRef<{ y: number; dy: number; moved: boolean; id: number } | null>(null);
  const draggedJustNow = useRef(false); // the click that ends a drag is not a tap on a tab
  const appEl = useRef<HTMLDivElement>(null);
  const topEl = useRef<HTMLElement>(null);
  // The panel pulled all the way up stops just under the search card, whose height varies.
  useEffect(() => {
    const top = topEl.current, app = appEl.current;
    if (!top || !app || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => app.style.setProperty("--top-h", `${Math.round(top.getBoundingClientRect().bottom)}px`));
    ro.observe(top);
    return () => ro.disconnect();
  }, [walking]); // the card is not shown while walking
  const track = useRef<Track | null>(null);
  const accuracy = useRef<number | null>(null);
  const request = useRef<AbortController | null>(null);
  const offCount = useRef(0);
  const showRoutes = useRef(true); // a new trip opens on its routes; changing a setting leaves the tab alone
  const colorMemory = useRef(new Map<string, string>());

  // Location stays in the browser; only the trip's two end points are sent for routing.
  useEffect(() => {
    if (!("geolocation" in navigator)) {
      setLocationNote("This browser does not share location. Search for a start or tap the map.");
      return;
    }
    const id = navigator.geolocation.watchPosition(
      (p) => { accuracy.current = p.coords.accuracy; setGps([p.coords.longitude, p.coords.latitude]); setLocationNote(null); },
      () => setLocationNote(t("Location is off. Search for a start or tap the map.")),
      { enableHighAccuracy: true, maximumAge: 5000 },
    );
    return () => navigator.geolocation.clearWatch(id);
  }, []);

  const user = demoAt ?? gps;
  const origin = pinnedOrigin?.at ?? user;
  const mix: Mix = [settings.cross, settings.slope, settings.sidewalk];

  // The barrier map follows the map view while its tab is open.
  useEffect(() => {
    if (tab !== "barriers" || !view) return;
    const ctl = new AbortController();
    const timer = setTimeout(() => {
      barriersIn(view, ctl.signal).then((b) => { setBarriers(b); setBarrierError(null); })
        .catch((e) => { if ((e as Error).name !== "AbortError") setBarrierError((e as Error).message); });
    }, 300);
    return () => { clearTimeout(timer); ctl.abort(); };
  }, [tab, view]);

  // `quiet` refreshes the numbers for new priorities while the current routes stay on
  // screen: they were planned under the same requirements, so they are still valid.
  async function find(from: LonLat | null = origin, opts: { keepWalking?: boolean; quiet?: boolean } = {}) {
    if (!from || !destination) return;
    request.current?.abort();
    const ctl = (request.current = new AbortController());
    setBusy(true); setError(null);
    if (!opts.quiet) setOpenPlace(null);
    try {
      const p = await planTrip(from, destination.at, latest.current, ctl.signal);
      if (request.current !== ctl) return; // the inputs changed while this was in flight: not their answer
      setPlan(p);
      setMixMoved(false);
      setSelected(p.route_options?.find((o) => o.is_recommended)?.id ?? p.route_options?.[0]?.id ?? null);
      if (showRoutes.current) { setTab("routes"); showRoutes.current = false; }
      if (!opts.keepWalking || p.status !== "ok") setWalking(false);
    } catch (e) {
      if ((e as Error).name !== "AbortError") { setError((e as Error).message); setPlan(null); setWalking(false); }
    } finally {
      if (request.current === ctl) setBusy(false);
    }
  }

  // Routes are planned as soon as there is a trip to plan, and again whenever a requirement
  // or an end of the trip changes: nobody should have to press a button to see their answer.
  // (Moving the point in the triangle needs no new plan; a walk in progress re-plans itself.)
  const haveOrigin = origin != null;
  useEffect(() => {
    tripToHash(pinnedOrigin, destination);
    if (!haveOrigin || !destination || walking) return;
    const timer = setTimeout(() => void find(), 250); // lets a slider settle
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [haveOrigin, pinnedOrigin, destination, settings.stepFree, settings.maxGrade, settings.certainty, settings.budgetMin]);

  useEffect(() => { showRoutes.current = true; }, [pinnedOrigin, destination]);

  function remember(e: End) {
    if (e.label === "Pinned on the map") return;
    const next = [e, ...recent.filter((r) => r.label !== e.label)].slice(0, 5);
    setRecent(next);
    save({ recent: next });
  }

  function swap() {
    if (!destination) return;
    clearPlan();
    const from = pinnedOrigin ?? (user ? { at: user, label: "My location" } : null);
    setPinnedOrigin(destination);
    setDestination(from);
  }

  // Any change to the trip's inputs goes through here. A request still in flight was asked
  // with the old inputs (perhaps without a requirement just switched on), so it is cancelled
  // and its answer can never be shown.
  function clearPlan() {
    request.current?.abort();
    request.current = null;
    setBusy(false);
    setPlan(null); setOpenPlace(null); setWalking(false); setDemoAt(null);
    colorMemory.current.clear();
  }

  function tap(p: LonLat) {
    if (walking) return;
    if (report && tab === "barriers") { setReport({ ...report, at: p, state: "draft" }); return; }
    clearPlan();
    if (tapSets === "origin") { setPinnedOrigin({ at: p, label: "Pinned on the map" }); setTapSets("destination"); }
    else setDestination({ at: p, label: "Pinned on the map" });
  }

  const all = useMemo(() => plan?.route_options ?? [], [plan]);
  // A route keeps its colour when the numbers are refreshed for new priorities, even though
  // the planner may list it under a different id: colours are remembered by the route's shape.
  const colors = useMemo(() => {
    const out: Record<number, string> = {};
    const memory = colorMemory.current;
    // the routes that win most of the triangle take the most distinct colours
    for (const o of [...all].sort((x, y) => y.priority_region_share - x.priority_region_share)) {
      const key = `${o.length_m}|${o.geometry.length}|${o.geometry[Math.floor(o.geometry.length / 2)]}`;
      if (!memory.has(key)) memory.set(key, o.is_fastest ? FASTEST_COLOR : ROUTE_COLORS[[...memory.values()].filter((c) => c !== FASTEST_COLOR).length % ROUTE_COLORS.length]);
      out[o.id] = memory.get(key)!;
    }
    return out;
  }, [all]);
  const winners = useMemo(() => {
    const m = new Map<string, number>();
    for (const [c, a, d, id] of plan?.triangle ?? []) m.set(gridKey([c, a, d]), id);
    return m;
  }, [plan]);
  // the chosen route, the fastest, then the routes that win most of the triangle
  const options = useMemo(() => {
    const lead = all.filter((o) => o.id === selected || o.is_fastest).sort((x, y) => Number(y.id === selected) - Number(x.id === selected));
    const rest = all.filter((o) => o.id !== selected && !o.is_fastest).sort((x, y) => y.priority_region_share - x.priority_region_share);
    return [...lead, ...rest.slice(0, 3)];
  }, [all, selected]);
  // How a route feels at the current mix, from the per-dimension minutes the planner sent:
  // the same sum the planner minimizes, so moving the point needs no round trip.
  const feltTotal = (o: RouteOption) =>
    o.minutes + 3 * (settings.cross * o.felt_minutes.cross + settings.slope * o.felt_minutes.slope + settings.sidewalk * o.felt_minutes.sidewalk);
  const fastest = all.find((o) => o.is_fastest);
  const feltSaved = (o: RouteOption) => (fastest ? feltTotal(fastest) - feltTotal(o) : 0);
  // Until the point is moved, the best fit is the planner's own answer for the exact mix;
  // after that it is the winner at the nearest point of the triangle's grid.
  const bestId = (mixMoved ? winners.get(gridKey(snap(mix))) : undefined) ?? all.find((o) => o.is_recommended)?.id ?? null;
  const chosen = all.find((o) => o.id === selected);
  const chosenPlaces = useMemo(() => places(chosen), [chosen]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const avoided = useMemo(() => (chosen && !chosen.is_fastest ? hazardsOf(chosenPlaces) : []), [chosenPlaces, chosen, language]);
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => { setSettings((s) => ({ ...s, [k]: v })); clearPlan(); };
  // A requirement changed by hand: the settings are no longer the profile's.
  const setRequirement = <K extends "stepFree" | "maxGrade" | "certainty">(k: K, v: Settings[K]) => {
    setSettings((s) => ({ ...s, [k]: v, profile: "custom" })); clearPlan();
  };
  function chooseProfile(p: Profile) {
    setSettings((s) => ({ ...s, profile: p.id, stepFree: p.requirements.step_free, maxGrade: p.requirements.max_grade_pct,
                          certainty: p.requirements.certainty, cross: p.mix.cross, slope: p.mix.slope, sidewalk: p.mix.sidewalk }));
    clearPlan();
  }
  const profile = profileList.find((p) => p.id === settings.profile);

  // Moving the point in the triangle changes only the mix of priorities. The routes already
  // planned cover every mix, so the winner at the new point is shown at once.
  function setMix(m: Mix) {
    setSettings((s) => ({ ...s, cross: m[0], slope: m[1], sidewalk: m[2] }));
    setMixMoved(true);
    const id = winners.get(gridKey(snap(m)));
    if (id !== undefined && id !== selected) { setSelected(id); setOpenPlace(null); }
  }

  // ---- walking: where the user is along the chosen route
  const line = chosen?.geometry ?? null;
  const cum = useMemo(() => (line ? cumulative(line) : null), [line]);
  const progress = useMemo(() => (walking && line && cum && user ? project(line, cum, user) : null), [walking, line, cum, user]);
  const total = cum ? cum[cum.length - 1] : 0;
  const remaining = progress ? Math.max(total - progress.along, 0) : total;
  const parts = useMemo(() => (walking && line && progress ? split(line, progress) : null), [walking, line, progress]);
  // Arrival means being at the end of the route, not merely abreast of it: a position far
  // off to the side can project onto the last few metres.
  const arrived = walking && progress != null && line != null && user != null
    && remaining <= ARRIVED_M && distance(user, line[line.length - 1]) <= ARRIVED_RADIUS_M;
  const nextPlace = useMemo(() => {
    if (!walking || !line || !cum || !progress) return null;
    let best: { index: number; ahead: number } | null = null;
    chosenPlaces.forEach((pl, index) => {
      const ahead = project(line, cum, pl.this_route.geometry[0]).along - progress.along;
      if (ahead > -10 && (best === null || ahead < best.ahead)) best = { index, ahead };
    });
    return best as { index: number; ahead: number } | null;
  }, [walking, line, cum, progress, chosenPlaces]);

  // the steps still ahead on the chosen route, nearest first
  const ahead = useMemo(() => (walking && cum && progress && chosen?.steps ? upcoming(chosen.steps, cum, progress.along) : []),
                        [walking, cum, progress, chosen]);
  const then = ahead[1] && ahead[1].step.type !== "arrive" && ahead[1].ahead - ahead[0].ahead <= THEN_WITHIN_M ? ahead[1] : null;
  const setOff = chosen?.steps?.[0]?.heading;

  // off the route twice running: plan again from where the user actually is
  useEffect(() => {
    if (!walking || !progress || !user || busy || arrived) return;
    if (progress.off <= OFF_ROUTE_M) { offCount.current = 0; return; }
    if (++offCount.current >= 2) {
      offCount.current = 0;
      track.current?.events.push({ t: Date.now(), type: "replanned", off_route_m: Math.round(progress.off) });
      setReplanned((n) => n + 1);
      setPinnedOrigin(null);
      void find(user, { keepWalking: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress]);

  // recording, when the walker has switched it on: one sample per position fix
  useEffect(() => {
    if (!walking || !recording || !progress || !user) return;
    const t = (track.current ??= { started: Date.now(), line: null, samples: [], events: [] });
    if (t.line !== line) { t.line = line; t.events.push({ t: Date.now(), type: "route", geometry: line ?? [] }); }
    t.samples.push({ t: Date.now(), lat: user[1], lon: user[0], accuracy_m: DEMO ? null : accuracy.current,
                     off_route_m: Math.round(progress.off * 10) / 10, along_m: Math.round(progress.along),
                     next_place_m: nextPlace ? Math.round(nextPlace.ahead) : null,
                     next_step_m: ahead[0] ? Math.round(ahead[0].ahead) : null });
    setRecorded(t.samples.length);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress]);
  useEffect(() => { if (arrived) track.current?.events.push({ t: Date.now(), type: "arrived" }); }, [arrived]);

  function saveRecording() {
    const t = track.current;
    if (!t) return;
    const body = { format: "pathfinder-walk-1", started: new Date(t.started).toISOString(), simulated: DEMO,
                   thresholds_m: { off_route: OFF_ROUTE_M, arrived: ARRIVED_M, arrived_radius: ARRIVED_RADIUS_M, next_place_alert: NEXT_PLACE_ALERT_M },
                   samples: t.samples, events: t.events };
    const url = URL.createObjectURL(new Blob([JSON.stringify(body)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `walk-${new Date(t.started).toISOString().slice(0, 16).replace(/[:T]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  function discardRecording() { track.current = null; setRecorded(0); }
  function stopWalking() {
    setWalking(false); setDemoAt(null); setTab("routes");
    if (feedback === "closed") setFeedback("after-walk");
  }

  // demo: a stand-in position at the trip's start, then walking the chosen route
  useEffect(() => {
    if (!DEMO) return;
    if (!walking) { setDemoAt(pinnedOrigin?.at ?? null); return; }
    if (!line || !cum) return;
    const total = cum[cum.length - 1];
    const step = (total * DEMO_TICK_MS) / (DEMO_ROUTE_S * 1000);
    let along = 0;
    setDemoAt(line[0]);
    const timer = setInterval(() => {
      along = Math.min(along + step, total);
      setDemoAt(pointAt(line, cum, along));
      if (along >= total) clearInterval(timer);
    }, DEMO_TICK_MS);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [walking, selected, plan, pinnedOrigin?.at?.[0], pinnedOrigin?.at?.[1]]);

  // a phone that sleeps mid-walk is no guide: hold the screen on while walking, where the browser allows
  useEffect(() => {
    if (!walking || !("wakeLock" in navigator)) return;
    let lock: WakeLockSentinel | null = null;
    const hold = () => navigator.wakeLock.request("screen").then((l) => { lock = l; }).catch(() => { /* denied or unsupported: carry on */ });
    void hold();
    const onVisible = () => { if (document.visibilityState === "visible") void hold(); };
    document.addEventListener("visibilitychange", onVisible);
    return () => { document.removeEventListener("visibilitychange", onVisible); void lock?.release(); };
  }, [walking]);

  function startWalking() {
    offCount.current = 0; setReplanned(0); setOpenPlace(null); discardRecording(); setWalking(true);
  }

  const ready = Boolean(origin && destination);

  // The panel follows a finger dragged on its handle or its tabs, and settles at the nearest
  // height in the direction it was moved. A tap on the handle still toggles it (keyboard too).
  function dragStart(e: React.PointerEvent) {
    if (e.pointerType === "mouse" || window.innerWidth >= 760) return;
    drag.current = { y: e.clientY, dy: 0, moved: false, id: e.pointerId };
  }
  function dragMove(e: React.PointerEvent) {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    d.dy = e.clientY - d.y;
    if (!d.moved && Math.abs(d.dy) < DRAG_START_PX) return;
    if (!d.moved) { d.moved = true; (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); }
    const el = sheetEl.current;
    if (el) { el.style.transition = "none"; el.style.transform = `translateY(${Math.max(d.dy, -40)}px)`; }
  }
  function dragEnd() {
    const d = drag.current;
    drag.current = null;
    const el = sheetEl.current;
    if (el) { el.style.transition = ""; el.style.transform = ""; }
    if (!d || !d.moved) return;
    draggedJustNow.current = true;
    setTimeout(() => { draggedJustNow.current = false; }, 0);
    const i = SHEET_ORDER.indexOf(sheet);
    if (d.dy <= -DRAG_SNAP_PX) setSheet(SHEET_ORDER[Math.min(i + 1, 2)]);
    else if (d.dy >= DRAG_SNAP_PX) setSheet(SHEET_ORDER[Math.max(i - (d.dy > 240 ? 2 : 1), 0)]);
  }
  const locate = user ? (
    <button className={walking && !following ? "locate wide-label" : "locate"} onClick={() => setRecenter((n) => n + 1)}
            aria-label={walking && !following ? t("Follow my position again") : t("Show my position")}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3" /><circle cx="12" cy="12" r="8" fill="none" /></svg>
      {walking && !following && <span>{t("Re-center")}</span>}
    </button>
  ) : null;
  const requirementCount = Number(settings.stepFree) + Number(settings.maxGrade != null) + Number(settings.certainty !== "allow_likely");

  return (
    <div ref={appEl} className={dark ? "app dark" : "app"}>
      <MapBoundary onRetry={() => setMapView(loadMap)}>
      <Suspense fallback={<div className="map" />}>
      {!painted ? <div className="map" /> : <MapView origin={walking ? null : (pinnedOrigin?.at ?? null)} destination={destination?.at ?? null} user={user}
               options={walking ? [] : options} extraRoute={plan?.relaxation?.route.geometry ?? null}
               selectedId={selected} places={chosenPlaces} openPlace={openPlace} colors={colors} dark={dark}
               walked={parts?.behind ?? null} ahead={parts?.ahead ?? null} follow={walking} layout={walking ? "walking" : sheet}
               followMs={DEMO ? DEMO_TICK_MS + 30 : undefined}
               hazards={walking ? [] : avoided}
               recenter={recenter} onFollowChange={setFollowing}
               report={tab === "barriers" ? report?.at ?? null : null}
               mode={tab === "barriers" && !walking && !report ? "barriers" : "routes"} barriers={barriers}
               onBarrier={(b) => { setBarrier(b); if (b) setSheet((s) => (s === "low" ? "mid" : s)); }} onView={setView}
               onTap={tap} onPlace={(i) => { setOpenPlace(i); setTab("routes"); setSheet((s) => (s === "low" ? "mid" : s)); }} />}
      </Suspense>
      </MapBoundary>

      {/* no box of its own, except on a phone held sideways: then it stacks the card and the panel down one side */}
      <div className="side" role="main">
      {!walking && (
        <header ref={topEl} className={destination ? "top compact" : "top"}>
          <div className="brand"><span className="mark" aria-hidden="true" />PathFinder
            {home || placeLinks.length ? <PlaceMenu current={townName} places={placeLinks} home={home} lang={language} />
                  : <small>{townName || t("curbs, slopes and sidewalks, as recorded")}</small>}
            <button className="lang" onClick={() => setLang(language === "es" ? "en" : "es")} lang={language === "es" ? "en" : "es"}
                    aria-label={language === "es" ? "Switch to English" : "Cambiar a español"}>{language === "es" ? "EN" : "ES"}</button>
          </div>
          <SearchBox kind="start" placeholder={t("Where from?")} near={user}
                     label={pinnedOrigin ? t(pinnedOrigin.label) : user ? t("My location") : ""}
                     active={tapSets === "origin"} onFocus={() => setTapSets("origin")}
                     recent={recent}
                     onPick={(at, label) => { clearPlan(); setPinnedOrigin({ at, label }); remember({ at, label }); setTapSets("destination"); }}
                     onMyLocation={gps ? () => { clearPlan(); setPinnedOrigin(null); setTapSets("destination"); } : undefined} />
          <button className="swap" onClick={swap} disabled={!destination} aria-label={t("Swap start and destination")}>⇅</button>
          <SearchBox kind="finish" placeholder={t("Where to?")} near={origin}
                     label={destination ? t(destination.label) : ""} active={tapSets === "destination"} onFocus={() => setTapSets("destination")}
                     recent={recent}
                     onPick={(at, label) => { clearPlan(); setDestination({ at, label }); remember({ at, label }); }} />
          {!online && <p className="hint offline">{t("You are offline. The map and routes need a connection.")}</p>}
          {locationNote && !pinnedOrigin && <p className="hint">{locationNote}</p>}
          {error && online && <p className="error">{error}</p>}
        </header>
      )}

      {walking && chosen ? (
        <section className="panel walking" style={{ "--route": colors[chosen.id] } as React.CSSProperties}>
          {locate}
          {/* the content scrolls inside, so the locate button can float above the panel */}
          <div className="walk-body">
          {arrived ? <h2 className="big">{t("You have arrived")}</h2> : (
            <>
              {ahead[0] ? (
                <>
                  <NextStep next={ahead[0]} then={then} />
                  {setOff && progress && progress.along < START_HINT_M && <p className="hint">{t("Set off heading {dir}.", { dir: setOff })}</p>}
                  <p className="togo"><b>{metres(remaining)}</b> · {t("about {m} to go", { m: min((chosen.minutes * remaining) / Math.max(total, 1)) })} · {t("arrive {time}", { time: clock((chosen.minutes * remaining) / Math.max(total, 1)) })}</p>
                </>
              ) : (
                <h2 className="big">{metres(remaining)} <small>{t("about {m} to go", { m: min((chosen.minutes * remaining) / Math.max(total, 1)) })} · {t("arrive {time}", { time: clock((chosen.minutes * remaining) / Math.max(total, 1)) })}</small></h2>
              )}
              <div className="progress" aria-hidden="true"><i style={{ width: `${100 * (1 - remaining / Math.max(total, 1))}%` }} /></div>
              {progress && progress.off > OFF_ROUTE_M && <p className="callout warn">{t("You are {d} from the route. Finding a new one…", { d: metres(progress.off) })}</p>}
              {replanned > 0 && <p className="hint">{t("Route updated from where you are.")}</p>}
              {nextPlace && nextPlace.ahead <= NEXT_PLACE_ALERT_M
                ? <button className="callout" onClick={() => setOpenPlace(nextPlace.index)}>
                    <b>{nextPlace.index + 1}</b> {nextPlace.ahead > 15
                      ? t("Decision point in {d}: your route leaves the fastest one.", { d: metres(nextPlace.ahead) })
                      : t("Decision point here: your route leaves the fastest one.")} <u>{t("See why")}</u>
                  </button>
                : nextPlace && <p className="hint">{t("Next decision point in {d}.", { d: metres(nextPlace.ahead) })}</p>}
            </>
          )}
          {openPlace != null && chosenPlaces[openPlace] && <PlaceCard place={chosenPlaces[openPlace]} index={openPlace} onClose={() => setOpenPlace(null)} />}
          <label className="switch record">
            <input type="checkbox" checked={recording} onChange={(e) => { setRecording(e.target.checked); save({ recordWalks: e.target.checked }); }} />
            <span>{t("Record this walk")} <small>{recording ? t("{n} positions, kept on this phone", { n: recorded }) : t("to help tune guidance")}</small></span>
          </label>
          <button className="btn ghost wide" onClick={stopWalking}>{t(arrived ? "Done" : "Stop")}</button>
          {DEMO && <p className="hint">{t("Demo: the position is simulated along the route.")}</p>}
          </div>
        </section>
      ) : (
        <section ref={sheetEl} className={`panel ${sheet}`}>
          {locate}
          <div className="handle" onPointerDown={dragStart} onPointerMove={dragMove} onPointerUp={dragEnd} onPointerCancel={dragEnd}
               onClickCapture={(e) => { if (draggedJustNow.current) { e.stopPropagation(); e.preventDefault(); } }}>
          <button className="grip" aria-expanded={sheet !== "low"} aria-label={sheet === "low" ? "Show the panel" : "Show more of the map"}
                  onClick={() => setSheet((s) => (s === "low" ? "mid" : "low"))} />
          <nav className="tabs" role="tablist">
            <button role="tab" aria-selected={tab === "priorities"} onClick={() => setTab("priorities")}>
              {t("What matters")}{requirementCount > 0 && <em>{requirementCount}</em>}
            </button>
            <button role="tab" aria-selected={tab === "routes"} onClick={() => setTab("routes")}>{t("Routes")}{all.length > 0 && <em>{all.length}</em>}</button>
            <button role="tab" aria-selected={tab === "barriers"} onClick={() => setTab("barriers")}>{t("Barriers")}</button>
          </nav>
          </div>

          {tab === "barriers" ? (
            report ? <ReportPane draft={report} onChange={setReport} onDone={() => setReport(null)}
                                 centre={view ? [(view[0] + view[2]) / 2, (view[1] + view[3]) / 2] : null} />
                   : <BarrierPane barriers={barriers} error={barrierError} chosen={barrier} onClose={() => setBarrier(null)}
                                  onReport={() => { setBarrier(null); setReport({ at: null, kind: null, note: "", state: "draft" }); }} />
          ) : tab === "priorities" ? (
            <div className="pane">
              <h3>{t("How I get around")}</h3>
              <p className="hint profile-prompt">{t("Tap the one that fits you. You can fine-tune it below.")}</p>
              <div className="profiles" role="radiogroup" aria-label={t("How I get around")}>
                {profileList.map((p) => (
                  <button key={p.id} role="radio" aria-checked={settings.profile === p.id}
                          className={settings.profile === p.id ? "chip on" : "chip"} onClick={() => chooseProfile(p)}>{tProfile(p.id, p.label)}</button>
                ))}
              </div>
              {profile && <p className="hint profile-note">{tProfile(profile.id, profile.description, "description")}</p>}
              <Triangle mix={mix} onChange={setMix} winners={plan?.status === "ok" ? winners : undefined} colors={colors} chosen={selected} />
              <p className="hint center">
                {plan?.status === "ok" && chosen
                  ? <>{t("Each colour is a route. At this mix:")} <b style={{ color: colors[chosen.id] }}>{chosen.is_fastest ? t("the fastest route") : t("{m}, {extra} longer", { m: min(chosen.minutes), extra: min(chosen.compared_with_fastest?.extra_minutes ?? 0) })}</b></>
                  : t("Drag the point toward what matters most on this walk.")}
              </p>

              <div className="row">
                <label className="budget">{t("Up to {n} min longer", { n: settings.budgetMin })}
                  <input type="range" min={0} max={15} step={1} value={settings.budgetMin} onChange={(e) => set("budgetMin", Number(e.target.value))} />
                </label>
              </div>

              <h3>{t("Must have")} <small>{t("never traded away")}</small></h3>
              <div className="chips">
                <button className={settings.stepFree ? "chip on" : "chip"} aria-pressed={settings.stepFree} onClick={() => setRequirement("stepFree", !settings.stepFree)}>{t("Step-free")}</button>
                {SLOPES.map(([v, label]) => (
                  <button key={label} className={settings.maxGrade === v ? "chip on" : "chip"} aria-pressed={settings.maxGrade === v} onClick={() => setRequirement("maxGrade", v)}>{t(label)}</button>
                ))}
              </div>
              <h3>{t("Evidence I accept")}</h3>
              <div className="chips segmented">
                {CERTAINTY.map(([v, label]) => (
                  <button key={v} className={settings.certainty === v ? "chip on" : "chip"} aria-pressed={settings.certainty === v} onClick={() => setRequirement("certainty", v)}>{t(label)}</button>
                ))}
              </div>
            </div>
          ) : (
            <div className="pane">
              {recorded > 0 && (
                <div className="place saved-walk">
                  <h4>Walk recorded<button className="link" onClick={discardRecording}>discard</button></h4>
                  <p className="hint">{recorded} positions with their distance from the route. Nothing has left this phone; saving makes a file you can pass on yourself.</p>
                  <button className="btn ghost wide" onClick={saveRecording}>Save recording</button>
                </div>
              )}
              {feedback !== "closed" && <Feedback walked={feedback === "after-walk"} onClose={() => setFeedback("closed")} />}
              {!plan && busy && <div className="skeleton" aria-label={t("Finding routes")}><i /><i /><i /></div>}
              {!plan && !busy && <p className="empty">{ready ? t("Ready when you are.") : t("Choose where from and where to, by searching above or tapping the map.")}</p>}
              {!plan && !busy && !destination && examples.length > 0 && (
                <div className="examples">
                  <h3>{t("Try an example")}</h3>
                  {examples.map((e) => (
                    <button key={e.label} className="example" onClick={() => {
                      const p = profileList.find((x) => x.id === e.profile);
                      if (p) {
                        setSettings((s) => ({ ...s, profile: p.id, stepFree: p.requirements.step_free, maxGrade: p.requirements.max_grade_pct,
                                              certainty: e.certainty ?? p.requirements.certainty, cross: p.mix.cross, slope: p.mix.slope, sidewalk: p.mix.sidewalk }));
                      }
                      clearPlan();
                      setPinnedOrigin({ at: [e.from[0], e.from[1]], label: e.from[2] });
                      setDestination({ at: [e.to[0], e.to[1]], label: e.to[2] });
                      setTapSets("destination");
                    }}>
                      <b>{(language === "es" && e.label_es) || e.label}</b>{e.note && <span>{(language === "es" && e.note_es) || e.note}</span>}
                    </button>
                  ))}
                </div>
              )}
              {plan && <Results plan={plan} options={options} total={all.length} selected={selected} colors={colors} bestId={bestId}
                                feltTotal={feltTotal} feltSaved={feltSaved}
                                onSelect={(id) => { setSelected(id); setOpenPlace(null); }}
                                places={chosenPlaces} openPlace={openPlace} onPlace={setOpenPlace} />}
              {plan && feedback === "closed" && <button className="link block" onClick={() => setFeedback("open")}>{t("Tell us how this is working")}</button>}
            </div>
          )}

          {/* what a screen reader hears when routes arrive */}
          <p className="sr-only" role="status" aria-live="polite">{busy ? t("Finding routes…") : !plan ? ""
            : plan.status === "ok" ? plural(all.length, "{n} route found.", "{n} routes found.") + (chosen ? " " + t("Selected: {name}, {m}.", { name: t(chosen.id === bestId ? "Best fit" : chosen.is_fastest ? "Fastest" : "Alternative"), m: min(chosen.minutes) }) : "")
            : plan.status === "no_feasible_route" ? t("No route meets everything you asked for.") : plan.message ?? ""}</p>
          <footer className="cta">
            {plan?.status === "ok" && chosen
              ? <button className="btn primary" style={{ background: colors[chosen.id], color: textOn(colors[chosen.id]) }} onClick={startWalking}>
                  {DEMO ? t("Start walking (demo)") : t("Start walking")} · {min(chosen.minutes)} · {t("arrive {time}", { time: clock(chosen.minutes) })}
                </button>
              : <button className="btn primary" disabled={!ready || busy} onClick={() => find()}>
                  {busy ? t("Finding routes…") : !destination ? t("Where to?") : !origin ? t("Where from?") : t("Find routes")}
                </button>}
            {plan?.status === "ok" && busy && <span className="spinner" aria-label="Updating" />}
          </footer>
        </section>
      )}
      </div>
    </div>
  );
}

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** The town name in the header opens a menu of every place, so another city is one click away
 *  (each place has its own server, so a choice opens that server's page). */
function PlaceMenu({ current, places, home, lang }: { current: string; places: PlaceLink[]; home: string | null; lang: string }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const q = new URLSearchParams();
  if (lang === "es") q.set("lang", "es");
  if (DEMO) q.set("demo", "");
  const suffix = q.toString() ? `?${q.toString().replace(/=$|=(?=&)/g, "")}` : "";
  useEffect(() => {
    if (!open) return;
    const away = (e: Event) => { if (!box.current?.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") { setOpen(false); box.current?.querySelector("button")?.focus(); } };
    document.addEventListener("pointerdown", away); document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("pointerdown", away); document.removeEventListener("keydown", esc); };
  }, [open]);
  const here = (name: string) => current.startsWith(name.split(",")[0]);
  return (
    <div className="place-menu" ref={box}>
      <button className="town" aria-haspopup="true" aria-expanded={open} onClick={() => setOpen((o) => !o)}
              title={t("Choose another place")}>
        {current || t("Change place")} <span aria-hidden="true">▾</span>
      </button>
      {open && (
        <nav className="place-list" aria-label={t("Choose another place")}>
          {places.map((p) => (
            <a key={p.url} href={p.url + suffix} aria-current={here(p.name) ? "page" : undefined}
               className={here(p.name) ? "here" : undefined}>{p.name}{here(p.name) && <small>{t("you are here")}</small>}</a>
          ))}
          {home && <a className="all" href={home + suffix}>{t("All places")}</a>}
        </nav>
      )}
    </div>
  );
}

/** Text colour for a label on a route-coloured button: white or ink, whichever reads better. */
function textOn(hex: string): string {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
  const l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  return 1.05 / (l + 0.05) >= (l + 0.05) / (0.0099 + 0.05) ? "#ffffff" : "#101a2b";
}

interface ReportDraft { at: LonLat | null; kind: ReportKind | null; note: string; state: "draft" | "sending" | "sent" | "failed" }
const REPORT_KINDS: [ReportKind, string][] = [["broken_ramp", "Broken curb ramp"], ["no_ramp", "No curb ramp"], ["blocked_sidewalk", "Sidewalk blocked"],
                                               ["no_sidewalk", "No sidewalk"], ["too_steep", "Too steep"], ["other", "Other"]];

/** Report a barrier: tap the map, say what is there, send. It is kept by this server for the team
 *  to check in person and changes nothing on the map until then. */
function ReportPane({ draft, onChange, onDone, centre }: { draft: ReportDraft; onChange: (d: ReportDraft) => void; onDone: () => void; centre: LonLat | null }) {
  async function send() {
    if (!draft.at || !draft.kind) return;
    onChange({ ...draft, state: "sending" });
    try {
      await sendReport(draft.at, draft.kind, draft.note);
      onChange({ ...draft, state: "sent" });
    } catch {
      onChange({ ...draft, state: "failed" });
    }
  }
  if (draft.state === "sent") {
    return (
      <div className="pane report">
        <div className="place" role="status">
          <h4>{t("Thank you. Your report is saved for the team to check in person.")}</h4>
          <p className="hint">{t("Nothing changes on the map until someone has checked it on site. Only the place, what you chose and today's date are kept.")}</p>
        </div>
        <div className="row-buttons">
          <button className="btn ghost" onClick={() => onChange({ at: null, kind: null, note: "", state: "draft" })}>{t("Report another")}</button>
          <button className="btn ghost" onClick={onDone}>{t("close")}</button>
        </div>
      </div>
    );
  }
  return (
    <div className="pane report">
      <h3>{t("Report a barrier")}</h3>
      <p className={draft.at ? "hint" : "hint callout"} aria-live="polite">{draft.at ? `✓ ${draft.at[1].toFixed(5)}, ${draft.at[0].toFixed(5)}` : t("Tap the map where the barrier is.")}</p>
      {centre && <button className="link block" onClick={() => onChange({ ...draft, at: centre })}>{t("Or use the centre of the map")}</button>}
      <h3>{t("What is there?")}</h3>
      <div className="chips" role="radiogroup" aria-label={t("What is there?")}>
        {REPORT_KINDS.map(([k, label]) => (
          <button key={k} role="radio" aria-checked={draft.kind === k} className={draft.kind === k ? "chip on" : "chip"}
                  onClick={() => onChange({ ...draft, kind: k })}>{t(label)}</button>
        ))}
      </div>
      <textarea className="report-note" aria-label={t("Anything else? (optional)")} placeholder={t("Anything else? (optional)")} maxLength={300} rows={2}
                value={draft.note} onChange={(e) => onChange({ ...draft, note: e.target.value })} />
      {draft.state === "failed" && <p className="error" role="alert">{t("That did not go through. Try again in a moment.")}</p>}
      <div className="row-buttons">
        <button className="btn primary" disabled={!draft.at || !draft.kind || draft.state === "sending"} onClick={() => void send()}>{t("Send report")}</button>
        <button className="btn ghost" onClick={onDone}>{t("Cancel")}</button>
      </div>
      <p className="fine">{t("Nothing changes on the map until someone has checked it on site. Only the place, what you chose and today's date are kept.")}</p>
    </div>
  );
}

function PlaceCard({ place, index, onClose }: { place: Place; index: number; onClose: () => void }) {
  return (
    <div className="place">
      <h4><b className="num">{index + 1}</b>{t("Decision point")}<button className="link" onClick={onClose}>{t("close")}</button></h4>
      <p className="lede">{t("{extra} longer here, and it feels {felt} easier.", { extra: min(place.extra_minutes), felt: min(place.net_felt_minutes_saved) })}</p>
      <div className="versus">
        <div><h5 className="this">{t("Your route · {len} m", { len: place.this_route.length_m })}</h5><Conditions items={place.this_route.conditions} /></div>
        <div><h5 className="fastest">{t("Fastest · {len} m", { len: place.fastest_route.length_m })}</h5><Conditions items={place.fastest_route.conditions} /></div>
      </div>
    </div>
  );
}

interface ResultsProps {
  plan: Plan; options: RouteOption[]; total: number; selected: number | null; colors: Record<number, string>; bestId: number | null;
  feltTotal: (o: RouteOption) => number; feltSaved: (o: RouteOption) => number;
  onSelect: (id: number) => void; places: Place[]; openPlace: number | null; onPlace: (i: number | null) => void;
}

/** The planner's message for a trip end too far from any walkway, translated when it is the usual one. */
function farMessage(message: string): string {
  const m = /^There is no mapped walkway within (\d+) m of the (origin|destination)\.$/.exec(message);
  if (!m) return message;
  return t(m[2] === "origin" ? "There is no mapped walkway within {d} m of the start." : "There is no mapped walkway within {d} m of the destination.", { d: m[1] });
}

function Results({ plan, options, total, selected, colors, bestId, feltTotal, feltSaved, onSelect, places, openPlace, onPlace }: ResultsProps) {
  if (plan.status === "not_near_network") return <p className="empty">{farMessage(plan.message ?? "")}</p>;
  if (plan.status === "no_feasible_route") {
    const r = plan.relaxation;
    return (
      <div className="results">
        <p className="callout warn"><b>!</b> {t("No route meets everything you asked for.")}</p>
        {r ? (
          <div className="place">
            <h4>{t("Smallest compromise")}</h4>
            <p className="lede">{t(r.edges_outside_original_requirements === 1 ? "{what}: {m}, {len}, with {n} stretch outside what you asked for." : "{what}: {m}, {len}, with {n} stretches outside what you asked for.",
                                   { what: capital(t(r.compromise)), m: min(r.route.minutes), len: metres(r.route.length_m), n: r.edges_outside_original_requirements })}</p>
            <h5>{t("Where you would be accepting it")}</h5>
            <Conditions items={r.where_it_applies} />
          </div>
        ) : <p className="hint">{plan.message}</p>}
      </div>
    );
  }
  const scale = Math.max(...options.map(feltTotal), 1);
  const dims = DIMS;
  const open = openPlace != null ? places[openPlace] : null;
  return (
    <div className="results">
      {(plan.transit_notices ?? []).map((n, i) => (
        <p key={i} className="callout warn">
          <b>!</b> {n.station} Metro, at your {n.end === "origin" ? "start" : "destination"}: {n.status === "unknown" ? n.message
            : <>an elevator is out of service{n.where ? ` (${n.where.replace(/^Elevator /i, "")})` : ""}{n.estimated_return_eastern ? `, expected back ${new Date(n.estimated_return_eastern).toLocaleDateString([], { month: "short", day: "numeric" })}` : ""}.</>}
          <small className="fine"> {n.source}{n.as_of ? `, read ${new Date(n.as_of).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}` : ""}</small>
        </p>
      ))}
      {(["origin", "destination"] as const).map((end) => plan[end]?.inside_closed_area && (
        <p key={end} className="callout warn"><b>!</b> OpenStreetMap shows your {end === "origin" ? "start" : "destination"} inside {plan[end]!.inside_closed_area}, which is not open to the public. The route {end === "origin" ? "starts" : "ends"} at the nearest public walkway outside it.</p>
      ))}
      {plan.destination?.nearer_walkway_passed_over && <p className="hint">The nearest walkway to your destination does not meet your requirements; the route ends at the nearest one that does.</p>}
      {options.map((o) => {
        const c = o.compared_with_fastest;
        return (
          <button key={o.id} className={o.id === selected ? "route chosen" : "route"} style={{ "--route": colors[o.id] } as React.CSSProperties} onClick={() => onSelect(o.id)}>
            <span className="swatch" />
            <span className="name">{t(o.id === bestId ? (o.is_fastest ? "Fastest · best fit" : "Best fit") : o.is_fastest ? "Fastest" : "Alternative")}
              <small>{metres(o.length_m)}{c ? ` · ${plural(c.places_that_differ, "differs in {n} place", "differs in {n} places")}` : ""}</small></span>
            <span className="time">{Math.round(o.minutes)}<small>min</small></span>
            <span className="bar" aria-label={t("{walk} of walking plus {felt} of felt effort", { walk: min(o.minutes), felt: min(feltTotal(o) - o.minutes) })}>
              <i className="walk" style={{ width: `${(100 * o.minutes) / scale}%` }} />
              {dims.map(([k, , color]) => <i key={k} style={{ width: `${(100 * (o.felt_minutes[k] ?? 0)) / scale}%`, background: color }} />)}
            </span>
            {c && <span className="delta">{t("+{extra} walking", { extra: min(c.extra_minutes) })}{feltSaved(o) > 0.05 ? t(", feels {m} easier", { m: min(feltSaved(o)) }) : feltSaved(o) < -0.05 ? t(", feels {m} harder at this mix", { m: min(-feltSaved(o)) }) : ""} · {t("arrive {time}", { time: clock(o.minutes) })}</span>}
            {!c && <span className="delta">{t("arrive {time}", { time: clock(o.minutes) })}</span>}
          </button>
        );
      })}
      <p className="legend"><span><i className="walk" />{t("walking")}</span>{dims.map(([k, label, color]) => <span key={k}><i style={{ background: color }} />{t(label.toLowerCase())}</span>)}</p>
      {total > options.length && <p className="hint">{plural(total - options.length, "{n} more route wins elsewhere in the triangle. Drag the point under “What matters” to reach them.", "{n} more routes win elsewhere in the triangle. Drag the point under “What matters” to reach them.")}</p>}
      {plan.note && <p className="hint">{plan.note}</p>}
      {places.length > 0 && !open && <p className="hint">{t("Numbered points on the map are where this route leaves the fastest one. Tap one to see why.")}</p>}
      {open && <PlaceCard place={open} index={openPlace!} onClose={() => onPlace(null)} />}
      <p className="fine">{plan.disclaimer}</p>
    </div>
  );
}

function BarrierPane({ barriers, error, chosen, onClose, onReport }: { barriers: Barriers | null; error: string | null; chosen: BarrierProps | null; onClose: () => void; onReport: () => void }) {
  return (
    <div className="pane">
      <button className="btn ghost wide report-start" onClick={onReport}>{t("Report a barrier")}</button>
      <p className="hint">{t("What the city's and OpenStreetMap's records show in this view: barriers in colour, and in grey the places nobody has recorded. Tap one to see its source and date.")}</p>
      {error && <p className="error">{error}</p>}
      <ul className="legend-list">
        {LEGEND.map((row) => (
          <li key={row.label} className={row.kind}>
            <i style={{ background: BARRIER_COLORS[row.types[0]] }} />{row.label}
            <b>{barriers ? row.types.reduce((n, t) => n + (barriers.counts[t] ?? 0), 0).toLocaleString() : "…"}</b>
          </li>
        ))}
      </ul>
      {barriers?.sidewalks && barriers.sidewalks.count > 0 && (
        <ul className="legend-list">
          <li className="evidence">
            <i style={{ background: ESTIMATED_SIDEWALK_COLOR }} />Sidewalk from a survey, line estimated
            <b>{barriers.sidewalks.count.toLocaleString()}</b>
          </li>
        </ul>
      )}
      {barriers?.truncated && <p className="hint">Counted in full, but too many to draw at once: zoom in to see every unrecorded place.</p>}
      {chosen && (
        <div className="place">
          <h4>{t(chosen.kind === "barrier" ? "Barrier" : chosen.kind === "evidence" ? "Sidewalk (estimated line)" : "Not recorded")}<button className="link" onClick={onClose}>{t("close")}</button></h4>
          <ul className="facts">
            <Fact text={`${chosen.what}${chosen.street ? ` · ${streetName(chosen.street)}` : ""}${chosen.length_m ? ` (${Math.round(chosen.length_m)} m)` : ""}`}
                  claim={{ status: chosen.status, source: chosen.source, as_of: chosen.as_of }} />
          </ul>
        </div>
      )}
      {barriers && barriers.unlicensed_sources.length > 0 && (
        <p className="fine internal">Internal: built in part on city data shared without an open license ({barriers.unlicensed_sources.join(", ")}). Not for use outside until the owners agree.</p>
      )}
      <p className="fine">{t("Nothing here certifies a place as accessible.")}</p>
    </div>
  );
}

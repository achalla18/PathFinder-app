import type { Barriers, LonLat, Plan, PlaceResult, Profile, Settings } from "./types";

/** A fetch that says plainly when the phone has no connection, instead of "Failed to fetch". */
async function get(url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new Error(navigator.onLine === false ? "You are offline. Routes need a connection." : "PathFinder could not be reached. Check the connection and try again.");
  }
}

export async function planTrip(origin: LonLat, destination: LonLat, s: Settings, signal?: AbortSignal): Promise<Plan> {
  const res = await get("/api/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      origin: { lon: origin[0], lat: origin[1] },
      destination: { lon: destination[0], lat: destination[1] },
      step_free: s.stepFree,
      max_grade_pct: s.maxGrade,
      certainty: s.certainty,
      budget_min: s.budgetMin,
      priorities: { cross: s.cross, slope: s.slope, sidewalk: s.sidewalk },
    }),
  });
  if (!res.ok) {
    throw new Error(res.status === 422 ? "Those settings are not valid." : `The route service answered ${res.status}.`);
  }
  return res.json();
}

export interface FeedbackAnswers {
  felt_minutes_clear?: "yes" | "partly" | "no";
  places_clear?: "yes" | "partly" | "no";
  better_than_route_list?: "yes" | "partly" | "no";
  walked: boolean;
  comment: string;
}

/** A barrier someone saw. Kept by this server for the team to check on site; nothing else is sent. */
export type ReportKind = "broken_ramp" | "no_ramp" | "blocked_sidewalk" | "no_sidewalk" | "too_steep" | "other";
export async function sendReport(at: LonLat, kind: ReportKind, note: string): Promise<void> {
  const res = await get("/api/report", { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ lon: at[0], lat: at[1], kind, note }) });
  if (!res.ok) throw new Error(`The service answered ${res.status}.`);
}

/** Send a tester's answers. Nothing else goes with them. */
export async function sendFeedback(answers: FeedbackAnswers): Promise<void> {
  const res = await get("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(answers) });
  if (!res.ok) throw new Error(`The service answered ${res.status}.`);
}

/** Whether this server has its own map tiles, how far they zoom, and where its region is.
 *  Asked once, as the page starts, so the answer is ready when the map's code arrives. */
let basemapAnswer: Promise<{ local: boolean; maxZoom: number; center: LonLat | null }> | null = null;
export function basemapInfo(): Promise<{ local: boolean; maxZoom: number; center: LonLat | null }> {
  return (basemapAnswer ??= askBasemap());
}
async function askBasemap(): Promise<{ local: boolean; maxZoom: number; center: LonLat | null }> {
  try {
    const h = await (await fetch("/api/health")).json();
    const center = Array.isArray(h.center) && h.center.length === 2 ? (h.center as LonLat) : null;
    return { local: Boolean(h.basemap), maxZoom: h.basemap_max_zoom ?? 15, center };
  } catch {
    return { local: false, maxZoom: 15, center: null };
  }
}

/** The mobility profiles this server offers. Kept on the phone after the first visit, so the
 *  buttons are there at once next time (the server's answer replaces them when it comes). */
const PROFILES_KEY = "pathfinder.profiles";
export function cachedProfiles(): Profile[] {
  try { return JSON.parse(localStorage.getItem(PROFILES_KEY) ?? "[]") as Profile[]; } catch { return []; }
}
export async function profiles(): Promise<Profile[]> {
  try {
    const list = ((await (await fetch("/api/region")).json()).profiles ?? []) as Profile[];
    try { localStorage.setItem(PROFILES_KEY, JSON.stringify(list)); } catch { /* private browsing */ }
    return list;
  } catch {
    return [];
  }
}

/** This server's town and the other towns' servers the page can switch to. */
/** A ready-made trip the page offers before anything is chosen: [lon, lat, name] at each end. */
export interface Example {
  label: string; note: string; from: [number, number, string]; to: [number, number, string];
  profile?: string; certainty?: "verified_only" | "allow_likely" | "allow_unknown";
  label_es?: string; note_es?: string;
}
export interface PlaceLink { name: string; url: string }
export async function townInfo(): Promise<{ name: string; home: string | null; places: PlaceLink[]; examples: Example[] }> {
  try {
    const r = await (await fetch("/api/region")).json();
    return { name: String(r.name ?? ""), home: typeof r.home === "string" ? r.home : null,
             places: Array.isArray(r.places) ? (r.places as PlaceLink[]) : [],
             examples: Array.isArray(r.examples) ? (r.examples as Example[]) : [] };
  } catch {
    return { name: "", home: null, places: [], examples: [] };
  }
}

/** Recorded barriers and unrecorded places in a map view. */
export async function barriersIn(bbox: [number, number, number, number], signal?: AbortSignal): Promise<Barriers> {
  const res = await get(`/api/barriers?bbox=${bbox.map((v) => v.toFixed(5)).join(",")}`, { signal });
  if (!res.ok) throw new Error(`The barrier map answered ${res.status}.`);
  return res.json();
}

/** Places and addresses matching what was typed, nearest first when a position is given. */
export async function searchPlaces(q: string, near: LonLat | null, signal?: AbortSignal): Promise<PlaceResult[]> {
  const params = new URLSearchParams({ q: q.trim() });
  if (near) { params.set("lon", String(near[0])); params.set("lat", String(near[1])); }
  const res = await fetch(`/api/search?${params}`, { signal });
  if (!res.ok) return [];
  return (await res.json()).results;
}

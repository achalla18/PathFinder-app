import { type GeoJSONSource, LngLatBounds, Map as MapLibreMap, type MapMouseEvent, Marker, prewarm, setWorkerCount, setWorkerUrl, type ExpressionSpecification, type StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre 6 ships its worker as a separate module; without it routes and pins (GeoJSON
// sources) never draw, while the raster basemap still does. Bundle it and point the library at it.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { layers as basemapLayers, namedFlavor } from "@protomaps/basemaps";
import { useEffect, useRef, useState } from "react";
import { basemapInfo } from "./api";
import { BARRIER_COLORS, ESTIMATED_SIDEWALK_COLOR } from "./barrierStyle";
import type { BarrierProps, Barriers, LonLat, Place, RouteOption } from "./types";
import type { Hazard } from "./hazards";

// The backdrop is drawn from this server's own vector tiles when it has them (`pf fetch
// basemap_dc basemap_fonts`), so no outside service sees where anyone is looking. The two
// flavours share layer ids: in the dark theme the paint of each layer is swapped, nothing reloads.
// Layers that need an icon sheet are left out; there is none to load.
const BASE = "protomaps";
const flavour = (name: string) => basemapLayers(BASE, namedFlavor(name), { lang: "en" }).filter((l) => !(l.layout && "icon-image" in l.layout));
const DAY = flavour("light");
const NIGHT = flavour("dark");
const localStyle = (maxzoom: number): StyleSpecification => ({
  version: 8,
  glyphs: `${window.location.origin}/fonts/{fontstack}/{range}.pbf`,
  sources: { [BASE]: { type: "vector", tiles: [`${window.location.origin}/tiles/{z}/{x}/{y}.mvt`], maxzoom, attribution: "© OpenStreetMap contributors · Protomaps" } },
  layers: DAY,
});

// Fallback when the server has no tiles of its own: OpenStreetMap's raster tiles, which are
// for light development use only.
const STYLE: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© OpenStreetMap contributors",
    },
  },
  // washed out on purpose: the map is the backdrop, the routes are the subject
  layers: [{ id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.85, "raster-contrast": -0.12, "raster-opacity": 0.9 } }],
};

setWorkerUrl(workerUrl);
// Each worker downloads and compiles its own copy of the worker script (half a megabyte). The
// library starts half the phone's cores' worth; two are plenty for a town's routes and tiles, and
// starting them now overlaps that work with the page's own start.
setWorkerCount(Math.min(2, Math.max(1, Math.floor((navigator.hardwareConcurrency || 2) / 2))));
prewarm();

// where the map opens when the server does not say which region it serves
const DC_CENTER: LonLat = [-77.0365, 38.9072];
const EMPTY = { type: "FeatureCollection" as const, features: [] };
const INK = "#101a2b";
const CORAL = "#e4572e";
const USER_BLUE = "#2f6df6";
const WIDE_PX = 760; // the width at which the panel sits beside the map (styles.css)
const PHONE_MARGIN_PX = 28; // room around a framed trip on a phone: a numbered pin is this wide
// A phone held sideways: the card and the panel are stacked down the left side (styles.css, .side)
const SIDE_MEDIA = "(orientation: landscape) and (max-height: 520px)";
const MIN_VIEW_PX = 80; // the least map height worth framing a trip into
const GESTURE_PX = 10; // a finger moving farther than this on the map is the user moving it, not a tap
const REDUCED_MOTION = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const DRAW_MS = 650; // how long a newly chosen route takes to draw itself in

/** A colour moved toward white (t > 0) or toward black (t < 0); t in [-1, 1]. */
function shade(hex: string, t: number): string {
  const n = parseInt(hex.slice(1), 16);
  const to = t > 0 ? 255 : 0;
  const k = Math.abs(t);
  const ch = (v: number) => Math.round(v + (to - v) * k);
  return `rgb(${ch((n >> 16) & 255)}, ${ch((n >> 8) & 255)}, ${ch(n & 255)})`;
}

/** A small white chevron, drawn once, repeated along the chosen route to show which way it runs. */
function chevron(): ImageData {
  const size = 32;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  g.strokeStyle = "#ffffff";
  g.lineWidth = 5;
  g.lineCap = g.lineJoin = "round";
  g.beginPath();
  g.moveTo(11, 8);
  g.lineTo(21, 16);
  g.lineTo(11, 24);
  g.stroke();
  return g.getImageData(0, 0, size, size);
}

interface Props {
  origin: LonLat | null;
  destination: LonLat | null;
  user: LonLat | null;
  options: RouteOption[];
  extraRoute: LonLat[] | null; // a compromise route when nothing meets the requirements
  selectedId: number | null;
  places: Place[];
  openPlace: number | null;
  walked: LonLat[] | null; // while walking: the part of the chosen route already covered
  ahead: LonLat[] | null; // ...and the part still to go
  follow: boolean; // keep the map centred on the user
  colors: Record<number, string>; // one colour per route option, shared with the cards and the triangle
  dark: boolean; // the dark theme (the phone's own setting): the map dims with the page
  layout: string; // changes when the panel takes more or less of the screen: the trip is framed again
  recenter: number; // bumped by the locate button: go to the user (and, while walking, follow again)
  onFollowChange?: (following: boolean) => void; // while walking: false once the user pans the map away
  report?: LonLat | null; // where a barrier report is being placed
  hazards?: Hazard[]; // what the chosen route avoids, labelled on the fastest route
  followMs?: number; // how long the map glides to each new position while following (default 800)
  mode: "routes" | "barriers"; // barriers: the city barrier map; a tap picks a barrier, not a destination
  barriers: Barriers | null;
  onTap: (p: LonLat) => void;
  onPlace: (index: number) => void;
  onBarrier: (b: BarrierProps | null) => void;
  onView: (bbox: [number, number, number, number]) => void; // the map came to rest here
}

export function MapView(props: Props) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const ready = useRef(false);
  const [styled, setStyled] = useState(false); // the style has loaded: sources exist and can be drawn
  const following = useRef(true); // while walking: the map keeps the user in view until they pan it
  const markers = useRef<Marker[]>([]);
  const hazardMarks = useRef<Marker[]>([]);
  const ends = useRef<Partial<Record<"origin" | "destination" | "user" | "report", Marker>>>({});
  const latest = useRef(props);
  latest.current = props;

  useEffect(() => {
    let gone = false;
    void basemapInfo().then(({ local, maxZoom, center }) => { if (!gone) build(local ? localStyle(maxZoom) : STYLE, center ?? DC_CENTER); });
    return () => { gone = true; map.current?.remove(); map.current = null; ready.current = false; setStyled(false); ends.current = {}; markers.current = []; hazardMarks.current = []; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function build(style: StyleSpecification, center: LonLat) {
    // pixelRatio: phone screens are 3x, but routes over washed-out tiles look the same at 2x, and
    // the map then draws under half as many pixels
    // the town comes into view with a short glide in (none for anyone who asks for less motion)
    const m = new MapLibreMap({ container: el.current!, style, center, zoom: REDUCED_MOTION ? 13 : 11.6, attributionControl: { compact: true },
                                pixelRatio: Math.min(window.devicePixelRatio || 1, 2), fadeDuration: 150 });
    map.current = m;
    // taps must work, and routes must draw, before the basemap tiles have arrived
    m.on("click", (e: MapMouseEvent) => {
      if (latest.current.mode === "barriers") {
        // a finger is wider than a curb point: look a few pixels around the tap
        const r = 10;
        const hit = ready.current ? m.queryRenderedFeatures([[e.point.x - r, e.point.y - r], [e.point.x + r, e.point.y + r]],
                                                           { layers: ["barrier-points", "barrier-lines", "estimated-sidewalks"] }) : [];
        latest.current.onBarrier(hit.length ? (hit[0].properties as BarrierProps) : null);
        return;
      }
      latest.current.onTap([e.lngLat.lng, e.lngLat.lat]);
    });
    const view = () => {
      const b = m.getBounds();
      latest.current.onView([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
    };
    m.on("moveend", view);
    // The user moving the map themselves while walking (a drag, a pinch, a wheel): stop pulling it
    // back to them. Watched on the finger itself, not on the map's move events: a position fix
    // eases the map about once a second, and a drag that starts during that animation joins a
    // move already under way, so the map reports no new start for it.
    const canvas = m.getCanvasContainer();
    let down: { x: number; y: number } | null = null;
    const letGo = () => {
      if (latest.current.follow && following.current) { following.current = false; latest.current.onFollowChange?.(false); }
    };
    canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; });
    canvas.addEventListener("pointermove", (e) => {
      if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > GESTURE_PX) { down = null; letGo(); }
    });
    for (const end of ["pointerup", "pointercancel"]) canvas.addEventListener(end, () => { down = null; });
    canvas.addEventListener("wheel", letGo, { passive: true });
    m.once("load", () => {
      view();
      if (!REDUCED_MOTION && latest.current.options.length === 0) m.easeTo({ zoom: 13, duration: 1800, easing: (t) => 1 - Math.pow(1 - t, 3) });
    });
    m.once("style.load", () => {
      for (const id of ["routes", "sides", "barriers", "sidewalks"]) m.addSource(id, { type: "geojson", data: EMPTY });
      // the chosen route and the stretch still to walk are drawn with a gradient along their length
      for (const id of ["chosen", "walked", "ahead"]) m.addSource(id, { type: "geojson", data: EMPTY, lineMetrics: true });
      m.addImage("chevron", chevron(), { pixelRatio: 2 });
      const round = { "line-cap": "round" as const, "line-join": "round" as const };
      const byZoom = (z13: number, z17: number) => ["interpolate", ["linear"], ["zoom"], 13, z13, 17, z17] as never;

      // The barrier map, under everything a trip draws.
      const barrierColor = ["match", ["get", "type"], ...Object.entries(BARRIER_COLORS).flat(), "#8a94a3"] as never;
      // zoom must be the outermost expression: the size for barriers and unknowns sits inside each stop
      const byKind = (b13: number, u13: number, b17: number, u17: number) =>
        ["interpolate", ["linear"], ["zoom"], 13, ["match", ["get", "kind"], "barrier", b13, u13],
         17, ["match", ["get", "kind"], "barrier", b17, u17]] as never;
      // estimated sidewalk lines: dashed, under the barriers (evidence, not a barrier)
      m.addLayer({ id: "estimated-sidewalks", type: "line", source: "sidewalks", layout: round,
        paint: { "line-color": ESTIMATED_SIDEWALK_COLOR, "line-width": byZoom(1.2, 3), "line-opacity": 0.8,
                 "line-dasharray": [1.5, 1.2] } });
      m.addLayer({ id: "barrier-lines", type: "line", source: "barriers", filter: ["==", ["geometry-type"], "LineString"], layout: round,
        paint: { "line-color": barrierColor, "line-width": byKind(3, 1.5, 6, 3.5),
                 "line-opacity": ["match", ["get", "kind"], "barrier", 0.9, 0.55] as never } });
      m.addLayer({ id: "barrier-points", type: "circle", source: "barriers", filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-color": barrierColor, "circle-radius": byKind(4, 2, 8, 4.5),
                 "circle-stroke-color": "#ffffff", "circle-stroke-width": ["match", ["get", "kind"], "barrier", 1.5, 0.5] as never } });

      // An open decision point: a soft band under both ways through it.
      m.addLayer({ id: "sides-band", type: "line", source: "sides", layout: round,
        paint: { "line-color": ["match", ["get", "side"], "this", INK, CORAL], "line-width": byZoom(16, 26), "line-opacity": 0.2, "line-blur": 3 } });

      // The other routes: present, in their own colours, but clearly in the background.
      m.addLayer({ id: "others-casing", type: "line", source: "routes", layout: round,
        paint: { "line-color": "#ffffff", "line-width": byZoom(4.5, 7), "line-opacity": 0.85 } });
      m.addLayer({ id: "others", type: "line", source: "routes", layout: round,
        paint: { "line-color": ["get", "color"], "line-width": byZoom(2.5, 4), "line-opacity": 0.8 } });

      // The chosen route: a glow in its colour, a white casing, then the line itself, lighter
      // where it starts and deepest where it ends, with chevrons along it.
      m.addLayer({ id: "chosen-glow", type: "line", source: "chosen", layout: round,
        paint: { "line-color": USER_BLUE, "line-width": byZoom(14, 24), "line-blur": 10, "line-opacity": 0.32 } });
      m.addLayer({ id: "chosen-casing", type: "line", source: "chosen", layout: round,
        paint: { "line-color": "#ffffff", "line-width": byZoom(8, 13) } });
      m.addLayer({ id: "chosen-line", type: "line", source: "chosen", layout: round,
        paint: { "line-color": USER_BLUE, "line-width": byZoom(5, 8.5) } });
      m.addLayer({ id: "chosen-arrows", type: "symbol", source: "chosen", minzoom: 13.5,
        layout: { "symbol-placement": "line", "symbol-spacing": 80, "icon-image": "chevron", "icon-size": byZoom(0.7, 1.05),
                  "icon-allow-overlap": true, "icon-ignore-placement": true },
        paint: { "icon-opacity": 0.92 } });

      // The fastest route's side of an open decision point, dashed over everything else.
      m.addLayer({ id: "side-fastest", type: "line", source: "sides", filter: ["==", ["get", "side"], "fastest"], layout: { "line-cap": "round" },
        paint: { "line-color": CORAL, "line-width": byZoom(3, 5), "line-dasharray": [0.4, 1.8] } });

      // Walking: what is behind is a faint dotted trail; what is ahead keeps the full treatment.
      m.addLayer({ id: "walked", type: "line", source: "walked", layout: { "line-cap": "round" },
        paint: { "line-color": "#8794a3", "line-width": byZoom(4, 6), "line-dasharray": [0.2, 1.7] } });
      m.addLayer({ id: "ahead-glow", type: "line", source: "ahead", layout: round,
        paint: { "line-color": USER_BLUE, "line-width": byZoom(16, 26), "line-blur": 10, "line-opacity": 0.34 } });
      m.addLayer({ id: "ahead-casing", type: "line", source: "ahead", layout: round,
        paint: { "line-color": "#ffffff", "line-width": byZoom(9, 14) } });
      m.addLayer({ id: "ahead-line", type: "line", source: "ahead", layout: round,
        paint: { "line-color": USER_BLUE, "line-width": byZoom(6, 9.5) } });
      m.addLayer({ id: "ahead-arrows", type: "symbol", source: "ahead", minzoom: 13.5,
        layout: { "symbol-placement": "line", "symbol-spacing": 70, "icon-image": "chevron", "icon-size": byZoom(0.75, 1.1),
                  "icon-allow-overlap": true, "icon-ignore-placement": true },
        paint: { "icon-opacity": 0.95 } });
      ready.current = true;
      dim();
      setStyled(true); // the drawing effects below run now, each for its own part
      fit();
    });
  }

  // Each part of the map is redrawn only when its own data changes. Redrawing everything on
  // every render of the page (each move in the triangle, each position fix while walking)
  // reset every line and rebuilt every pin.
  const src = (id: string, features: object[]) =>
    (map.current?.getSource(id) as GeoJSONSource | undefined)?.setData({ type: "FeatureCollection", features } as never);
  const line = (coordinates: LonLat[], properties: object = {}) =>
    ({ type: "Feature" as const, properties, geometry: { type: "LineString" as const, coordinates } });
  // a line's own colour, running from a lighter start to a deeper end
  const gradient = (color: string): ExpressionSpecification =>
    ["interpolate", ["linear"], ["line-progress"], 0, shade(color, 0.42), 0.55, color, 1, shade(color, -0.22)];
  const paint = (prefix: string, color: string) => {
    const m = map.current!;
    m.setPaintProperty(`${prefix}-glow`, "line-color", color);
    m.setPaintProperty(`${prefix}-line`, "line-gradient", gradient(color));
  };
  // A newly chosen route draws itself from start to end, over its white casing.
  const drawing = useRef(0);
  const drawIn = (prefix: string, color: string) => {
    cancelAnimationFrame(drawing.current);
    if (REDUCED_MOTION) { paint(prefix, color); return; }
    const m = map.current!;
    m.setPaintProperty(`${prefix}-glow`, "line-color", color);
    const t0 = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - t0) / DRAW_MS);
      const p = 1 - Math.pow(1 - t, 3);
      if (!map.current?.getLayer(`${prefix}-line`)) return;
      if (t >= 1) { paint(prefix, color); return; }
      m.setPaintProperty(`${prefix}-line`, "line-gradient", ["case", ["<=", ["line-progress"], p], gradient(color), "rgba(0,0,0,0)"] as ExpressionSpecification);
      drawing.current = requestAnimationFrame(step);
    };
    drawing.current = requestAnimationFrame(step);
  };
  const drawnRoute = useRef<unknown>(null);
  // the barrier map is about the city, not this trip: the trip steps aside while it is open
  const trip = props.mode !== "barriers";
  const color = (props.selectedId != null && props.colors[props.selectedId]) || USER_BLUE;

  useEffect(() => {
    if (!styled) return;
    const p = latest.current;
    const chosen = trip ? p.options.find((o) => o.id === p.selectedId) : undefined;
    src("routes", trip ? p.options.filter((o) => o !== chosen).map((o) => line(o.geometry, { color: p.colors[o.id] ?? INK })) : []);
    // a compromise route (nothing met the requirements) is shown as the chosen line, in coral
    const main = chosen?.geometry ?? (trip ? p.extraRoute : null);
    src("chosen", main ? [line(main)] : []);
    if (main) {
      if (main !== drawnRoute.current) drawIn("chosen", chosen ? color : CORAL);
      else paint("chosen", chosen ? color : CORAL);
    }
    drawnRoute.current = main;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [styled, trip, props.options, props.selectedId, props.extraRoute, props.colors, color]);

  useEffect(() => {
    if (!styled) return;
    const b = props.mode === "barriers" ? props.barriers : null;
    src("barriers", b ? b.features : []);
    src("sidewalks", b?.sidewalks ? b.sidewalks.features : []);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [styled, props.mode, props.barriers]);

  useEffect(() => {
    if (!styled) return;
    src("walked", props.walked && props.walked.length > 1 ? [line(props.walked)] : []);
    src("ahead", props.ahead && props.ahead.length > 1 ? [line(props.ahead)] : []);
    if (props.ahead) paint("ahead", color);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [styled, props.walked, props.ahead, color]);

  useEffect(() => {
    if (!styled) return;
    const open = trip && props.openPlace != null ? props.places[props.openPlace] : null;
    src("sides", open ? [line(open.fastest_route.geometry, { side: "fastest" }), line(open.this_route.geometry, { side: "this" })] : []);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [styled, trip, props.openPlace, props.places]);

  // Numbered pins for the places where the chosen route leaves the fastest one. Plain HTML
  // markers: the basemap style carries no fonts for map-drawn text. Rebuilt only for a new
  // set of places; opening one only changes its look.
  useEffect(() => {
    const m = map.current;
    if (!styled || !m) return;
    markers.current.forEach((mk) => mk.remove());
    markers.current = (trip ? props.places : []).map((pl, i) => {
      const pin = document.createElement("button");
      pin.className = "place-pin";
      pin.textContent = String(i + 1);
      pin.setAttribute("aria-label", `Decision point ${i + 1}, where the routes differ`);
      pin.addEventListener("click", (ev) => { ev.stopPropagation(); latest.current.onPlace(i); });
      return new Marker({ element: pin }).setLngLat(pl.at).addTo(m);
    });
  }, [styled, trip, props.places]);
  // What the chosen route avoids: a label on the fastest route's stretch at each place
  const hazardKey = (trip ? props.hazards ?? [] : []).map((h) => `${h.kind}@${h.at[0].toFixed(5)},${h.at[1].toFixed(5)}:${h.label}`).join("|");
  useEffect(() => {
    const m = map.current;
    if (!styled || !m) return;
    hazardMarks.current.forEach((mk) => mk.remove());
    hazardMarks.current = (trip ? props.hazards ?? [] : []).map((h) => {
      // the map places the outer element with a transform; the tag's pop-in animates its own, inside
      const holder = document.createElement("div");
      holder.setAttribute("aria-hidden", "true"); // the same facts are in the decision-point cards
      holder.innerHTML = `<div class="hazard-tag ${h.kind}"><i></i><span></span></div>`;
      holder.querySelector("span")!.textContent = h.label;
      return new Marker({ element: holder, anchor: "bottom", offset: [0, -6] }).setLngLat(h.at).addTo(m);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [styled, hazardKey]);

  useEffect(() => {
    markers.current.forEach((mk, i) => mk.getElement().classList.toggle("open", i === props.openPlace));
  }, [styled, trip, props.places, props.openPlace]);

  // the ends of the trip and the user's position: moved, not rebuilt, on each fix (CSS makes the user pulse)
  useEffect(() => {
    const m = map.current;
    if (!styled || !m) return;
    for (const kind of ["origin", "destination", "user", "report"] as const) {
      const at = latest.current[kind] ?? null;
      const existing = ends.current[kind];
      if (!at) { existing?.remove(); delete ends.current[kind]; continue; }
      if (existing) { existing.setLngLat(at); continue; }
      const dot = document.createElement("div");
      dot.className = `end-pin ${kind}`;
      dot.setAttribute("aria-hidden", "true");
      ends.current[kind] = new Marker({ element: dot }).setLngLat(at).addTo(m);
    }
  }, [styled, props.origin?.[0], props.origin?.[1], props.destination?.[0], props.destination?.[1], props.user?.[0], props.user?.[1],
      props.report?.[0], props.report?.[1]]);

  // frame the trip when a new set of routes arrives
  const fitKey = props.options.map((o) => o.geometry.length).join(",") + (props.extraRoute?.length ?? "");
  function fit() {
    const m = map.current;
    const p = latest.current;
    const coords = [...p.options.flatMap((o) => o.geometry), ...(p.extraRoute ?? [])];
    if (!m || coords.length === 0) return;
    const b = coords.reduce((acc, c) => acc.extend(c), new LngLatBounds(coords[0], coords[0]));
    // keep the trip clear of the panel: beside the map on wide screens, over its lower part on phones
    const wide = window.innerWidth >= WIDE_PX;
    const side = sideCovered();
    const { top, bottom } = covered();
    const padding = side != null ? { top: PHONE_MARGIN_PX, bottom: PHONE_MARGIN_PX, left: side + PHONE_MARGIN_PX, right: PHONE_MARGIN_PX }
                  : wide ? { top: 60, bottom: 60, left: 440, right: 60 }
                         : { top: top + PHONE_MARGIN_PX, bottom: bottom + PHONE_MARGIN_PX, left: 40, right: 40 };
    m.fitBounds(b, { padding, maxZoom: 17, duration: REDUCED_MOTION ? 0 : 1100 });
  }

  // On a phone held sideways: how far the column of card and panel reaches into the map from
  // the left, measured like covered(); null in every other layout.
  function sideCovered(): number | null {
    if (!window.matchMedia(SIDE_MEDIA).matches) return null;
    const mapBox = el.current?.getBoundingClientRect();
    const side = el.current?.parentElement?.querySelector(".side")?.getBoundingClientRect();
    if (!mapBox || !side) return null;
    // never ask for more room than the map has, as in covered()
    return Math.max(Math.min(side.right - mapBox.left, mapBox.width - 2 * PHONE_MARGIN_PX - MIN_VIEW_PX), 0);
  }

  // turning the phone changes the layout: frame the trip again for the new one
  useEffect(() => {
    const q = window.matchMedia(SIDE_MEDIA);
    const turned = () => { map.current?.resize(); fit(); }; // the map may not have noticed its new size yet
    q.addEventListener("change", turned);
    return () => q.removeEventListener("change", turned);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // On a phone the search card lies over the top of the map and the panel over the bottom.
  // How much each takes depends on the phone and on what the panel is showing, so it is
  // measured: a guess left the ends of a north-south trip hidden under both.
  function covered(): { top: number; bottom: number } {
    const app = el.current?.parentElement;
    const mapBox = el.current?.getBoundingClientRect();
    if (!app || !mapBox) return { top: 0, bottom: 0 };
    const card = app.querySelector(".top")?.getBoundingClientRect();
    const panel = app.querySelector(".panel")?.getBoundingClientRect();
    const top = card ? Math.max(card.bottom - mapBox.top, 0) : 0;
    const bottom = panel ? Math.max(mapBox.bottom - panel.top, 0) : 0;
    // never ask for more room than the map has: MapLibre then refuses to move at all
    const spare = mapBox.height - 2 * PHONE_MARGIN_PX - MIN_VIEW_PX;
    return top + bottom <= spare ? { top, bottom } : { top: 0, bottom: Math.max(Math.min(bottom, spare), 0) };
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(fit, [fitKey, props.layout]);

  // in the dark theme the map itself goes dim with the page; the routes glow a little more against it
  function dim() {
    const m = map.current;
    if (!m || !ready.current) return; // applied again when the style has loaded
    const night = latest.current.dark;
    if (m.getLayer("osm")) {
      m.setPaintProperty("osm", "raster-brightness-max", night ? 0.42 : 1);
      m.setPaintProperty("osm", "raster-saturation", night ? -0.95 : -0.85);
    } else {
      for (const layer of night ? NIGHT : DAY) {
        for (const [k, v] of Object.entries(layer.paint ?? {})) if (m.getLayer(layer.id)) m.setPaintProperty(layer.id, k as never, v as never);
      }
    }
    for (const id of ["chosen-glow", "ahead-glow"]) if (m.getLayer(id)) m.setPaintProperty(id, "line-opacity", night ? 0.55 : 0.32);
  }
  useEffect(dim, [props.dark]);

  // the user's position in view, clear of the panel (and on phones of the card above)
  function goToUser(duration: number, linear = false) {
    const m = map.current;
    const u = latest.current.user;
    if (!m || !u) return;
    const wide = window.innerWidth >= WIDE_PX;
    const side = sideCovered();
    const { top, bottom } = covered();
    m.easeTo({ center: u, zoom: Math.max(m.getZoom(), 16.5), duration, ...(linear ? { easing: (t: number) => t } : {}),
               padding: side != null ? { left: side, top: 0, right: 0, bottom: 0 }
                      : wide ? { left: 420, top: 0, right: 0, bottom: 0 } : { left: 0, top, right: 0, bottom } });
  }
  // while walking the map follows the user, until they pan it themselves
  useEffect(() => { following.current = true; latest.current.onFollowChange?.(true); }, [props.follow]);
  const at = props.follow && props.user ? `${props.user[0].toFixed(5)},${props.user[1].toFixed(5)}` : null;
  useEffect(() => {
    if (at && following.current) goToUser(props.followMs ?? 800, props.followMs != null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [at]);
  // the locate button: back to the user and, while walking, following again
  useEffect(() => {
    if (!props.recenter) return;
    if (props.follow && !following.current) { following.current = true; latest.current.onFollowChange?.(true); }
    goToUser(600);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.recenter]);

  return <div ref={el} className="map" />;
}

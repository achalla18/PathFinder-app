import type { LonLat } from "./types";

// Small planar helpers. Over the few kilometres of a walk in one city an equirectangular
// projection around the point of interest is accurate to well under a metre.

const R = 6371008.8;
const rad = (d: number) => (d * Math.PI) / 180;

function toXY(p: LonLat, lat0: number): [number, number] {
  return [rad(p[0]) * Math.cos(rad(lat0)) * R, rad(p[1]) * R];
}

export function distance(a: LonLat, b: LonLat): number {
  const [ax, ay] = toXY(a, a[1]);
  const [bx, by] = toXY(b, a[1]);
  return Math.hypot(bx - ax, by - ay);
}

/** Cumulative distance along a line, one entry per vertex. */
export function cumulative(line: LonLat[]): number[] {
  const out = [0];
  for (let i = 1; i < line.length; i++) out.push(out[i - 1] + distance(line[i - 1], line[i]));
  return out;
}

export interface Projection {
  along: number; // metres from the start of the line to the nearest point on it
  off: number; // metres from the given point to that nearest point
  point: LonLat;
  segment: number; // index of the segment's first vertex
}

/** Nearest point on a line to p. */
export function project(line: LonLat[], cum: number[], p: LonLat): Projection {
  let best: Projection = { along: 0, off: Infinity, point: line[0], segment: 0 };
  const [px, py] = toXY(p, p[1]);
  for (let i = 0; i < line.length - 1; i++) {
    const [ax, ay] = toXY(line[i], p[1]);
    const [bx, by] = toXY(line[i + 1], p[1]);
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
    const off = Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    if (off < best.off) {
      const point: LonLat = [line[i][0] + t * (line[i + 1][0] - line[i][0]), line[i][1] + t * (line[i + 1][1] - line[i][1])];
      best = { along: cum[i] + t * Math.sqrt(len2), off, point, segment: i };
    }
  }
  return best;
}

/** The line split at a projection: what is behind and what is ahead. */
export function split(line: LonLat[], at: Projection): { behind: LonLat[]; ahead: LonLat[] } {
  return { behind: [...line.slice(0, at.segment + 1), at.point], ahead: [at.point, ...line.slice(at.segment + 1)] };
}

/** A distance as a walker would say it: to the nearest 10 m, or in km. */
export const metres = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m / 10) * 10} m`);

/** The point a given distance along a line (clamped to its ends). */
export function pointAt(line: LonLat[], cum: number[], along: number): LonLat {
  if (along <= 0) return line[0];
  for (let i = 1; i < line.length; i++) {
    if (cum[i] >= along) {
      const t = (along - cum[i - 1]) / Math.max(cum[i] - cum[i - 1], 1e-9);
      return [line[i - 1][0] + t * (line[i][0] - line[i - 1][0]), line[i - 1][1] + t * (line[i][1] - line[i - 1][1])];
    }
  }
  return line[line.length - 1];
}

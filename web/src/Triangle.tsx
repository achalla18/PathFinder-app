import { useRef } from "react";
import { t } from "./i18n";

// The Tradeoff Triangle: one point sets the mix of the three priorities. Once routes are
// planned, the triangle is also a map of which route wins for every mix, so dragging the
// point switches routes instantly and shows how far the next change of answer is.

const N = 10; // the planner's grid: priorities in steps of 10 %
const W = 240, H = 214;
const TOP: [number, number] = [W / 2, 16]; // crossings and curbs
const LEFT: [number, number] = [18, H - 22]; // slope
const RIGHT: [number, number] = [W - 18, H - 22]; // sidewalk

export type Mix = [number, number, number]; // crossings, slope, sidewalk; sums to 1

function toXY([c, a, d]: Mix): [number, number] {
  return [c * TOP[0] + a * LEFT[0] + d * RIGHT[0], c * TOP[1] + a * LEFT[1] + d * RIGHT[1]];
}

function toMix(x: number, y: number): Mix {
  const det = (LEFT[1] - RIGHT[1]) * (TOP[0] - RIGHT[0]) + (RIGHT[0] - LEFT[0]) * (TOP[1] - RIGHT[1]);
  let c = ((LEFT[1] - RIGHT[1]) * (x - RIGHT[0]) + (RIGHT[0] - LEFT[0]) * (y - RIGHT[1])) / det;
  let a = ((RIGHT[1] - TOP[1]) * (x - RIGHT[0]) + (TOP[0] - RIGHT[0]) * (y - RIGHT[1])) / det;
  let d = 1 - c - a;
  // outside the triangle: pull back to the nearest edge
  c = Math.max(c, 0); a = Math.max(a, 0); d = Math.max(d, 0);
  const s = c + a + d || 1;
  return [c / s, a / s, d / s];
}

export const gridKey = (m: Mix) => `${Math.round(m[0] * N)},${Math.round(m[1] * N)}`;

/** The grid point nearest a mix (rounding each share alone can break the sum). */
export function snap(m: Mix): Mix {
  let c = Math.round(m[0] * N), a = Math.round(m[1] * N);
  if (c + a > N) { if (m[0] * N - c < m[1] * N - a) c = N - a; else a = N - c; }
  return [c / N, a / N, (N - c - a) / N];
}

interface Props {
  mix: Mix;
  onChange: (m: Mix) => void;
  onRelease?: () => void;
  winners?: Map<string, number>; // grid point -> route option id, once routes are planned
  colors?: Record<number, string>;
  chosen?: number | null;
}

export function Triangle({ mix, onChange, onRelease, winners, colors, chosen }: Props) {
  const svg = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);

  const fromEvent = (e: React.PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    onChange(toMix(((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H));
  };
  const nudge = (dc: number, da: number) => {
    const c = Math.min(Math.max(mix[0] + dc, 0), 1), a = Math.min(Math.max(mix[1] + da, 0), 1 - c);
    onChange([c, a, 1 - c - a]);
    onRelease?.();
  };

  // small triangles of the grid, coloured by the route that wins there
  const cells: { points: string; id: number }[] = [];
  if (winners && colors) {
    const at = (i: number, j: number) => winners.get(`${i},${j}`);
    const pt = (i: number, j: number) => toXY([i / N, j / N, (N - i - j) / N]).map((v) => v.toFixed(1)).join(",");
    const majority = (ids: (number | undefined)[]) => {
      const known = ids.filter((v): v is number => v !== undefined);
      return known.sort((x, y) => known.filter((v) => v === y).length - known.filter((v) => v === x).length)[0];
    };
    for (let i = 0; i < N; i++) {
      for (let j = 0; i + j < N; j++) {
        const up = majority([at(i, j), at(i + 1, j), at(i, j + 1)]);
        if (up !== undefined) cells.push({ points: `${pt(i, j)} ${pt(i + 1, j)} ${pt(i, j + 1)}`, id: up });
        if (i + j < N - 1) {
          const down = majority([at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)]);
          if (down !== undefined) cells.push({ points: `${pt(i + 1, j)} ${pt(i, j + 1)} ${pt(i + 1, j + 1)}`, id: down });
        }
      }
    }
  }
  const [x, y] = toXY(mix);
  const outline = `${TOP.join(",")} ${LEFT.join(",")} ${RIGHT.join(",")}`;
  const pct = mix.map((v) => Math.round(v * 100));

  return (
    <div className="triangle">
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="slider" tabIndex={0}
           aria-label={t("Priorities: drag to set the mix of crossings and curbs, slope, and sidewalk")}
           aria-valuetext={`Crossings and curbs ${pct[0]} percent, slope ${pct[1]} percent, sidewalk ${pct[2]} percent`}
           onPointerDown={(e) => { dragging.current = true; (e.target as Element).setPointerCapture?.(e.pointerId); fromEvent(e); }}
           onPointerMove={(e) => dragging.current && fromEvent(e)}
           onPointerUp={() => { dragging.current = false; onRelease?.(); }}
           onKeyDown={(e) => {
             const step = 1 / N;
             if (e.key === "ArrowUp") nudge(step, -step / 2);
             else if (e.key === "ArrowDown") nudge(-step, step / 2);
             else if (e.key === "ArrowLeft") nudge(0, step);
             else if (e.key === "ArrowRight") nudge(0, -step);
             else return;
             e.preventDefault();
           }}>
        <defs>
          <linearGradient id="tri-idle" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="var(--cross)" stopOpacity="0.28" />
            <stop offset="0.5" stopColor="var(--slope)" stopOpacity="0.22" />
            <stop offset="1" stopColor="var(--sidewalk)" stopOpacity="0.3" />
          </linearGradient>
          <clipPath id="tri-clip"><polygon points={outline} /></clipPath>
        </defs>
        <polygon points={outline} fill={cells.length ? "var(--surface-2)" : "url(#tri-idle)"} />
        <g clipPath="url(#tri-clip)">
          {cells.map((c, k) => (
            <polygon key={k} points={c.points} fill={colors![c.id]} stroke={colors![c.id]} strokeWidth="0.6"
                     opacity={chosen == null || c.id === chosen ? 0.92 : 0.38} />
          ))}
        </g>
        <polygon points={outline} fill="none" stroke="var(--line-strong)" strokeWidth="1.5" strokeLinejoin="round" />
        {/* guides from the point to each corner show the mix at a glance */}
        {[TOP, LEFT, RIGHT].map((p, k) => <line key={k} x1={x} y1={y} x2={p[0]} y2={p[1]} stroke="var(--ink)" strokeOpacity="0.18" strokeDasharray="2 3" />)}
        <circle cx={x} cy={y} r="13" fill="var(--ink)" fillOpacity="0.12" />
        <circle cx={x} cy={y} r="8.5" fill="#fff" stroke="var(--ink)" strokeWidth="3" />
      </svg>
      <span className="corner at-top"><i style={{ background: "var(--cross)" }} />{t("Crossings & curbs")} <b>{pct[0]}%</b></span>
      <span className="corner at-left"><i style={{ background: "var(--slope)" }} />{t("Slope")} <b>{pct[1]}%</b></span>
      <span className="corner at-right"><i style={{ background: "var(--sidewalk)" }} />{t("Sidewalk")} <b>{pct[2]}%</b></span>
    </div>
  );
}

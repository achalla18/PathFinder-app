import { useEffect, useRef, useState } from "react";
import { searchPlaces } from "./api";
import type { LonLat, PlaceResult } from "./types";
import { t } from "./i18n";

interface Props {
  kind: "start" | "finish";
  label: string; // what the field currently stands for, e.g. a chosen place or "My location"
  placeholder: string;
  near: LonLat | null;
  active: boolean; // map taps currently set this end
  onFocus: () => void;
  onPick: (p: LonLat, label: string) => void;
  onMyLocation?: () => void; // offered on the start field when the browser shares location
  recent?: { at: LonLat; label: string }[]; // places picked before on this device, offered before typing
}

/** A text field that searches places and addresses on our own server as you type. */
export function SearchBox({ kind, label, placeholder, near, active, onFocus, onPick, onMyLocation, recent = [] }: Props) {
  const [text, setText] = useState("");
  const [editing, setEditing] = useState(false);
  const [results, setResults] = useState<PlaceResult[]>([]);
  const [searched, setSearched] = useState(false);
  const request = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!editing || text.trim().length < 2) { setResults([]); setSearched(false); return; }
    const timer = setTimeout(async () => {
      request.current?.abort();
      const ctl = (request.current = new AbortController());
      try {
        setResults(await searchPlaces(text, near, ctl.signal));
        setSearched(true);
      } catch { /* superseded by a newer keystroke, or offline: keep what is shown */ }
    }, 220);
    return () => clearTimeout(timer);
  }, [text, editing, near]);

  const done = () => { setEditing(false); setText(""); setResults([]); };

  return (
    <div className={active ? "search active" : "search"}>
      <i className={`dot ${kind}`} />
      <input
        value={editing ? text : label}
        placeholder={placeholder}
        aria-label={kind === "start" ? "Start" : "Destination"}
        onFocus={() => { setEditing(true); setText(""); onFocus(); }}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => setTimeout(done, 150) /* let a tap on a suggestion land first */}
        autoComplete="off" enterKeyHint="search" autoCorrect="off" spellCheck={false}
      />
      {editing && text && (
        <button className="clear" aria-label={t("Clear")} onMouseDown={(e) => { e.preventDefault(); setText(""); }}>×</button>
      )}
      {editing && (
        <ul className="suggestions">
          {onMyLocation && <li><button onMouseDown={(e) => { e.preventDefault(); onMyLocation(); done(); }}>◎ My location</button></li>}
          {results.map((r, i) => (
            <li key={i}>
              <button onMouseDown={(e) => { e.preventDefault(); onPick([r.lon, r.lat], r.label); done(); }}>
                {r.label}<small>{r.detail}</small>
              </button>
            </li>
          ))}
          {searched && results.length === 0 && <li className="none">{t("No match. Try a street address or a place name, or tap the map.")}</li>}
          {!searched && recent.map((r, i) => (
            <li key={`recent-${i}`}>
              <button onMouseDown={(e) => { e.preventDefault(); onPick(r.at, r.label); done(); }}>{r.label}<small>recent</small></button>
            </li>
          ))}
          {!searched && <li className="none">{t("Type a place or address, or tap the map.")}</li>}
        </ul>
      )}
    </div>
  );
}

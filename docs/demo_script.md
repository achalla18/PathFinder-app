# Demo video script (Congressional App Challenge, MD-6)

About 3 minutes. Everything shown is built from open data only (rule 7): DC's open data, and the
public builds of Gaithersburg and Frederick.

## Before recording

1. `pf demo` (from `C:\dev\PathFinder`, with the venv's `pf.exe`). Wait for
   "PathFinder demo: http://127.0.0.1:8000". It opens the browser.
2. **Wait for "example trips ready"** in the `pf demo` window. It plans every example once at start-up,
   so each one opens instantly on camera (even the Zoo trip, which takes a few seconds to work out).
3. **Browser window:** 1920×1080 with no other tabs or bookmarks bar showing. Zoom at 100%.
   Extensions with toolbars or pop-ups should be hidden or off. Light theme reads best on video.
4. **Recording:** Windows: Win+Alt+R (Xbox Game Bar), or OBS at 1080p, 30 fps.

## Shots

| # | Time | On screen | Say (suggested) |
|---|---|---|---|
| 1 | 0:00–0:20 | Landing page; the cards rise in and the evidence bars fill (the number is said, not shown) | "In Frederick, PathFinder can give a wheelchair user a route for only 3 of every 100 trips with every curb recorded or likely passable. In Washington, 65. For people who use a wheelchair, a walker or a stroller, the fastest route often isn't one they can take." |
| 2 | 0:20–0:35 | Point at the three cards and the bars: DC 92% curb evidence, Frederick 17% | "It works anywhere the data exists. Washington publishes its sidewalks and curb ramps; in Maryland's 6th district, most of it has never been recorded." |
| 3 | 0:35–1:05 | Click **Washington, DC**. The map glides in. Click **Sheridan Circle to the West End Library**. The routes draw in, and a tag on the fastest route shows what the recommended one avoids ("No sidewalk"). | "Here's a manual-wheelchair trip. The fastest route runs along a street with no sidewalk; PathFinder flags it on the map and recommends a route a minute longer that avoids it." |
| 4 | 1:05–1:20 | Open **What matters**: the profiles, then drag the point in the triangle | "Everyone's needs differ. Pick how you get around, then drag toward what matters most: crossings, slope or sidewalks." |
| 5 | 1:20–1:45 | Back to **Routes**. Click **Woodley Park Metro to the National Zoo** | "And when no route meets your needs, PathFinder says so. This hill is steeper than an ADA ramp. It shows the smallest compromise, and exactly where, with the source: DC's 2024 LiDAR survey." |
| 6 | 1:45–2:00 | **Barriers** tab; pan the map a little; then **Report a barrier**: tap a corner, choose "No curb ramp", send | "Every curb, slope and sidewalk carries its source and date. And anyone can report a barrier they find; the team checks it in person before anything changes." |
| 7 | 2:00–2:30 | Click **Washington, DC ▾** (Change place), then **Frederick**. Click **City Hall to Nicodemus Athletic Complex** | "Now Frederick, in our district. The recommended route takes a minute longer but avoids the hardest crossings. Because curbs here aren't recorded, it shows what isn't known instead of guessing." |
| 8 | 2:30–2:45 | Click **Start walking** with `?demo` in the address (see below) | "Then step-by-step guidance along the way." |
| 8b | (optional, 10 s) | Turn on Windows Narrator (Ctrl+Win+Enter), click an example: it reads "13 routes found. Selected: Best fit, 23 min." | "And it works with a screen reader." |
| 8c | (optional, 10 s) | Click **ES** in the header: the whole app switches to Spanish | "Frederick has a large Spanish-speaking community, so PathFinder speaks Spanish too." |
| 9 | 2:45–3:00 | Back on the landing page (place menu → All places); hold on the three cards and their bars | "The gap can be closed. A community mapping campaign in OpenStreetMap can record Frederick's sidewalks street by street, and every tag makes PathFinder better for everyone." |

**Simulated walk (shot 8):** open the demo as `http://127.0.0.1:8000/?demo`. Every place then
simulates your position: the blue dot waits at the trip's start, and after **Start walking** it
walks the whole route in about 18 seconds, with the guidance and the map following it. The panel
says "Demo: the position is simulated along the route"; say so in the narration too. Without
`?demo` the app waits for real GPS.

## Facts to keep straight on camera

- **Headline:** of 276–300 sampled trips per place, routable by manual wheelchair (step-free, at most
  8.33% grade) with every curb recorded or likely passable:

  | Place | Trips routable |
  |---|---|
  | Frederick | 3.3% |
  | Gaithersburg | 2.7% |
  | Washington | 65.3% |

  Open-data builds only; `pf crux` → `reports/<region>/phase0/crux_summary.json`, scenario
  `stepfree_likely`. Say "can be routed with every curb recorded or likely passable"; do not say
  "can't be walked".
- **Accessibility:** axe-core finds no WCAG 2.1 A/AA violations on the landing page (English and
  Spanish) or in the app (empty, routes, What matters, Barriers, report form), as of 2026-10-22.
  A pass with a real screen reader (NVDA or Narrator) by a person is still to do.
- **Spanish:** the app's own text is translated. Turn-by-turn steps, the conditions of a stretch and
  map labels stay in English.
- **Reports:** saved only by the local server (`<data dir>/reports/barrier_reports.csv`). They reach
  the field sheet with `tools/field_checks/add_reports.py` and change nothing until a visit.
  **After recording,** delete the report made on camera
  (`~/.pathfinder/data/regions/<region>/reports/barrier_reports.csv`) so it never reaches the sheet.

- **DC evidence:** curb evidence 92%, slope measured 82%, sidewalk evidence 75%. From
  `reports/dc/phase0/coverage.md`.
- **Frederick evidence:** curbs 17%, slope 96%, sidewalks 42%. **Gaithersburg:** curbs 19%,
  slope 89%, sidewalks 47%. From the open-data builds only.
- **Frederick and Gaithersburg examples** use "Anything not ruled out" as the evidence setting.
  With "Recorded or likely" most Maryland trips have no route, because the curbs are unrecorded:
  that is the data gap.
- **Mapping campaign:** say it "can" close the gap; it has not started yet. The forum post is
  pending.
- **Do not show** the internal builds (`frederick`, `gaithersburg` without `_public`); `pf demo`
  never starts them.

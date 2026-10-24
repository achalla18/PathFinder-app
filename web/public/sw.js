// PathFinder's service worker: keeps the app itself on the phone, so it opens at once on later
// visits and still opens with a weak signal. It never keeps a trip, a search or a position:
// /api/plan, /api/search, /api/barriers and /api/feedback always go to the server and are not
// stored.
const VERSION = "pathfinder-v1";
const SHELL = ["/", "/manifest.webmanifest", "/icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// the network first, the copy on the phone when the network fails
async function networkFirst(request) {
  const cache = await caches.open(VERSION);
  try {
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch (e) {
    const kept = await cache.match(request, { ignoreSearch: request.mode === "navigate" });
    if (kept) return kept;
    throw e;
  }
}

// files whose names carry a hash of their content never change: the copy on the phone is used
async function cacheFirst(request) {
  const cache = await caches.open(VERSION);
  const kept = await cache.match(request);
  if (kept) return kept;
  const response = await fetch(request);
  if (response.ok) cache.put(request, response.clone());
  return response;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return; // map tiles from OSM: the browser's own cache
  if (request.mode === "navigate") return event.respondWith(networkFirst(new Request("/", { credentials: "same-origin" })));
  if (url.pathname.startsWith("/assets/") || url.pathname.startsWith("/fonts/")) return event.respondWith(cacheFirst(request));
  // what the region offers (profiles, map centre): shown from the phone when offline
  if (url.pathname === "/api/region" || url.pathname === "/api/health") return event.respondWith(networkFirst(request));
  if (SHELL.includes(url.pathname) || /^\/(icon|apple-touch-icon)/.test(url.pathname)) return event.respondWith(networkFirst(request));
  // everything else (trips, searches, barriers, tiles) goes to the server untouched
});

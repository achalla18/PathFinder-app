import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

// Keeps the app on the phone for later visits (public/sw.js). Browsers allow it only over HTTPS or
// on this machine; elsewhere the page simply loads from the server each time.
if (import.meta.env.PROD && "serviceWorker" in navigator && window.isSecureContext) {
  window.addEventListener("load", () => { void navigator.serviceWorker.register("/sw.js").catch(() => { /* not allowed here: carry on */ }); });
}

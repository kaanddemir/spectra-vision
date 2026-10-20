// Applies the saved or system colour theme before the first paint, so no wrong theme flashes.
// Keep in step with src/theme/theme.ts (storage key, values and attribute).
(function () {
  var preference = "system";
  try {
    var saved = window.localStorage.getItem("spectra.theme");
    if (saved === "light" || saved === "dark" || saved === "system") preference = saved;
  } catch (error) {
    // Storage can be unavailable; the system theme applies.
  }
  var dark = preference === "dark" ||
    (preference === "system" && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
})();

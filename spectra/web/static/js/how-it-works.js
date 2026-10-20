// Standalone /how-it-works page: open the full-screen detail popups from the
// pipeline stage cards and the compact topic menu. Independent of the app's
// controls.js — this page never loads the analysis UI.
(function () {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const menu = byId("doc-topic-menu-list");
  const menuToggle = document.querySelector("[data-doc-menu-toggle]");
  const infoPanel = byId("doc-color-info-panel");
  const infoToggle = document.querySelector("[data-doc-info-toggle]");

  function enhanceInfoTooltips() {
    document.querySelectorAll(".flow-node").forEach((node) => {
      const title = node.querySelector("h5");
      const description = title ? title.nextElementSibling : null;
      if (!title || !description || description.tagName !== "P" || title.querySelector(".flow-info-icon")) {
        return;
      }
      const tooltip = description.textContent.trim();
      if (!tooltip) return;
      const icon = document.createElement("span");
      icon.className = "flow-info-icon";
      icon.dataset.tooltip = tooltip;
      icon.title = tooltip;
      icon.tabIndex = 0;
      icon.setAttribute("role", "button");
      icon.setAttribute("aria-label", tooltip);
      icon.setAttribute("aria-expanded", "false");
      title.appendChild(icon);

      const route = node.dataset.route ? node.dataset.route.trim() : "";
      if (route && !title.querySelector(".flow-route-icon")) {
        const routeIcon = document.createElement("span");
        routeIcon.className = "flow-route-icon";
        routeIcon.dataset.tooltip = route;
        routeIcon.title = route;
        routeIcon.tabIndex = 0;
        routeIcon.setAttribute("role", "button");
        routeIcon.setAttribute("aria-label", route);
        routeIcon.setAttribute("aria-expanded", "false");
        title.appendChild(routeIcon);
      }
    });
  }

  function closeInfoTooltips(except = null) {
    document.querySelectorAll(".flow-info-icon.is-open, .flow-route-icon.is-open").forEach((icon) => {
      if (icon === except) return;
      icon.classList.remove("is-open");
      icon.setAttribute("aria-expanded", "false");
    });
  }

  function setNavActive(name) {
    document.querySelectorAll(".doc-menu-btn").forEach((btn) => {
      btn.classList.toggle("is-active", !!name && btn.dataset.docOpen === name);
    });
  }

  function setMenuOpen(open) {
    if (!menu || !menuToggle) return;
    menu.hidden = !open;
    menuToggle.setAttribute("aria-expanded", String(open));
    if (open) setInfoOpen(false);
  }

  function setInfoOpen(open) {
    if (!infoPanel || !infoToggle) return;
    infoPanel.hidden = !open;
    infoToggle.setAttribute("aria-expanded", String(open));
    if (open) setMenuOpen(false);
  }

  function openModal(name) {
    const modal = byId(`doc-modal-${name}`);
    if (!modal) return;
    setMenuOpen(false);
    setInfoOpen(false);
    modal.hidden = false;
    // Force reflow so the opening transition runs from the hidden state.
    void modal.offsetHeight;
    modal.classList.add("is-open");
    document.body.classList.add("doc-modal-open");
    setNavActive(name);
  }

  function closeModal(modal) {
    if (!modal) return;
    modal.classList.remove("is-open");
    const done = () => {
      modal.hidden = true;
      if (!document.querySelector(".doc-modal.is-open")) {
        document.body.classList.remove("doc-modal-open");
        setNavActive(null);
      }
    };
    // Hide after the transition; fall back immediately if reduced motion.
    setTimeout(done, 200);
  }

  function closeAll() {
    document.querySelectorAll(".doc-modal.is-open").forEach(closeModal);
  }

  document.addEventListener("click", (event) => {
    if (event.target.closest(".flow-info-icon, .flow-route-icon")) {
      event.preventDefault();
      event.stopPropagation();
      const icon = event.target.closest(".flow-info-icon, .flow-route-icon");
      const shouldOpen = !icon.classList.contains("is-open");
      closeInfoTooltips(icon);
      icon.classList.toggle("is-open", shouldOpen);
      icon.setAttribute("aria-expanded", String(shouldOpen));
      return;
    }
    closeInfoTooltips();
    const menuButton = event.target.closest("[data-doc-menu-toggle]");
    if (menuButton) {
      event.preventDefault();
      setMenuOpen(menu ? menu.hidden : false);
      return;
    }
    const infoButton = event.target.closest("[data-doc-info-toggle]");
    if (infoButton) {
      event.preventDefault();
      setInfoOpen(infoPanel ? infoPanel.hidden : false);
      return;
    }
    const opener = event.target.closest("[data-doc-open]");
    if (opener) {
      event.preventDefault();
      openModal(opener.dataset.docOpen);
      return;
    }
    const closer = event.target.closest("[data-doc-close]");
    if (closer) {
      event.preventDefault();
      closeModal(closer.closest(".doc-modal"));
      return;
    }
    if (menu && !menu.hidden && !event.target.closest(".doc-topic-menu")) {
      setMenuOpen(false);
    }
    if (infoPanel && !infoPanel.hidden && !event.target.closest(".doc-info-menu")) {
      setInfoOpen(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    closeInfoTooltips();
    if (infoPanel && !infoPanel.hidden) {
      setInfoOpen(false);
      if (infoToggle) infoToggle.focus();
      return;
    }
    if (menu && !menu.hidden) {
      setMenuOpen(false);
      if (menuToggle) menuToggle.focus();
      return;
    }
    closeAll();
  });

  enhanceInfoTooltips();
})();

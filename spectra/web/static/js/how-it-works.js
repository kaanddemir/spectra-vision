// Standalone /how-it-works page: open the full-screen detail popups from the
// pipeline stage cards and the compact topic menu. Independent of the app's
// controls.js — this page never loads the analysis UI.
(function () {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const menu = byId("doc-topic-menu-list");
  const menuToggle = document.querySelector("[data-doc-menu-toggle]");
  const outputToggle = document.querySelector("[data-doc-output-toggle]");

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
      icon.setAttribute("aria-label", tooltip);
      title.appendChild(icon);
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
  }

  function toggleOutputs() {
    const isHidden = document.body.classList.toggle("doc-hide-outputs");
    if (!outputToggle) return;
    outputToggle.setAttribute("aria-pressed", String(isHidden));
    const label = outputToggle.querySelector("span");
    if (label) label.textContent = isHidden ? "Show outputs" : "Hide outputs";
    outputToggle.setAttribute("aria-label", isHidden ? "Show outputs" : "Hide outputs");
  }

  function openModal(name) {
    const modal = byId(`doc-modal-${name}`);
    if (!modal) return;
    setMenuOpen(false);
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
    if (event.target.closest(".flow-info-icon")) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    const menuButton = event.target.closest("[data-doc-menu-toggle]");
    if (menuButton) {
      event.preventDefault();
      setMenuOpen(menu ? menu.hidden : false);
      return;
    }
    const outputButton = event.target.closest("[data-doc-output-toggle]");
    if (outputButton) {
      event.preventDefault();
      toggleOutputs();
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
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (menu && !menu.hidden) {
      setMenuOpen(false);
      if (menuToggle) menuToggle.focus();
      return;
    }
    closeAll();
  });

  enhanceInfoTooltips();
})();

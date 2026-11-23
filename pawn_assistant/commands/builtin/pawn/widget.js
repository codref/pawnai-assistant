/* Pawn plugin — "pawn-chat" transcript window and "keypad" unlock window. */
(function () {
  "use strict";

  const { el } = window.PawnWidgets;

  // ------------------------------------------------------------------ chat

  window.PawnWidgets.register("pawn-chat", {
    mount(body, data, api) {
      const root = el("div", "pc");
      const head = el("div", "pc-head");
      const badge = el("span", "pc-badge");
      const status = el("span", "pc-status");
      head.append(badge, status);
      const log = el("div", "pc-log");
      log.setAttribute("aria-live", "polite");
      root.append(head, log);
      body.appendChild(root);

      let d = data;

      function renderBadge() {
        const full = d.access === "full";
        root.classList.toggle("unlocked", full);
        badge.textContent = full ? "FULL ACCESS" : "CHAT ONLY";
        api.setStatus(full ? "UNLOCKED" : "LOCKED");
      }

      function render(changed) {
        renderBadge();
        if (!changed || "status" in changed) status.textContent = d.status || "";
        if (!changed || "turns" in changed) {
          log.replaceChildren();
          for (const t of d.turns || []) {
            const row = el("div", `pc-turn pc-${t.who === "you" ? "you" : "pawn"}`);
            row.append(el("b", null, t.who === "you" ? "YOU" : "PAWN"), el("span", null, t.text));
            log.appendChild(row);
          }
          log.scrollTop = log.scrollHeight;
        }
        if (changed && changed.access === "full") api.flash();
      }

      render();
      return {
        update(next, changed) {
          d = next;
          render(changed);
        },
        unmount() {},
      };
    },
  });

  // ------------------------------------------------------------------ keypad

  const KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "⌫", "0", "OK"];

  window.PawnWidgets.register("keypad", {
    mount(body, data, api) {
      const root = el("div", "kp");
      const display = el("div", "kp-display");
      display.setAttribute("aria-label", "code");
      const msg = el("div", "kp-msg", "ENTER CODE");
      const grid = el("div", "kp-grid");
      root.append(display, msg, grid);
      body.appendChild(root);

      let code = "";
      let d = data;
      let lastNonce = d.nonce || 0;
      const maxLen = Math.max(4, Math.min(12, d.length || 8));

      function paint() {
        display.replaceChildren();
        const slots = Math.max(d.length || 4, code.length);
        for (let i = 0; i < slots; i += 1) {
          display.appendChild(el("i", i < code.length ? "on" : null));
        }
      }

      function press(key) {
        if (root.classList.contains("locked")) return;
        if (key === "⌫") {
          code = code.slice(0, -1);
        } else if (key === "OK") {
          if (!code) return;
          api.action("submit", { code });
          code = "";
          msg.textContent = "CHECKING…";
        } else if (code.length < maxLen) {
          code += key;
          msg.textContent = "ENTER CODE";
          root.classList.remove("wrong");
        }
        paint();
      }

      for (const key of KEYS) {
        const b = el("button", `hud-btn kp-key${key === "OK" ? " kp-ok" : ""}`, key);
        b.type = "button";
        b.addEventListener("pointerdown", (ev) => {
          ev.preventDefault(); // no focus ring / double-tap zoom on touch screens
          press(key);
        });
        grid.appendChild(b);
      }

      function onKey(ev) {
        if (/^[0-9]$/.test(ev.key)) press(ev.key);
        else if (ev.key === "Backspace") press("⌫");
        else if (ev.key === "Enter") press("OK");
        else return;
        ev.preventDefault();
        ev.stopPropagation();
      }
      document.addEventListener("keydown", onKey, true);

      function showResult() {
        if ((d.nonce || 0) === lastNonce) return;
        lastNonce = d.nonce || 0;
        root.classList.remove("wrong");
        void root.offsetWidth; // restart the shake animation
        if (d.result === "wrong") {
          root.classList.add("wrong");
          msg.textContent = "WRONG CODE";
          api.flash();
        } else if (d.result === "locked") {
          root.classList.add("locked");
          msg.textContent = "LOCKED";
        }
      }

      paint();
      return {
        update(next) {
          d = next;
          showResult();
          paint();
        },
        unmount() {
          document.removeEventListener("keydown", onKey, true);
          code = "";
        },
      };
    },
  });
})();

/* Timer plugin — "countdown" widget. Ticks locally from the server's due_at_ms. */
(function () {
  "use strict";

  const { el } = window.PawnWidgets;
  const SEGMENTS = 24;

  function fmt(ms) {
    const total = Math.max(0, Math.ceil(ms / 1000));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const mm = String(m).padStart(2, "0");
    const ss = String(s).padStart(2, "0");
    return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  }

  window.PawnWidgets.register("countdown", {
    mount(body, data, api) {
      const root = el("div", "cd");
      const digits = el("div", "cd-digits", "--:--");
      digits.setAttribute("role", "timer");
      digits.setAttribute("aria-live", "off");
      const label = el("div", "cd-label");
      const bar = el("div", "cd-bar");
      bar.setAttribute("aria-hidden", "true");
      const segs = [];
      for (let i = 0; i < SEGMENTS; i += 1) {
        const seg = el("i");
        bar.appendChild(seg);
        segs.push(seg);
      }
      const foot = el("div", "cd-foot");
      const state = el("span", "cd-state", "RUNNING");
      const cancel = el("button", "hud-btn danger", "CANCEL");
      cancel.type = "button";
      cancel.addEventListener("click", () => api.action("cancel"));
      foot.append(state, cancel);
      root.append(digits, label, bar, foot);
      body.appendChild(root);

      let d = data;
      let expiredShown = false;

      function tick() {
        const total = Math.max(1, (d.total_s || 1) * 1000);
        const left = d.state === "expired" ? 0 : Math.max(0, d.due_at_ms - api.now());
        const text = fmt(left);
        if (digits.textContent !== text) digits.textContent = text;
        api.setStatus(d.state === "expired" ? "EXPIRED" : text);
        const lit = Math.ceil((left / total) * SEGMENTS);
        segs.forEach((s, i) => s.classList.toggle("on", i < lit));
        root.classList.toggle("low", left > 0 && left <= 10_000);
        if (d.state === "expired" && !expiredShown) {
          expiredShown = true;
          root.classList.add("expired");
          state.textContent = "EXPIRED";
          cancel.textContent = "DISMISS";
          api.flash();
        }
      }

      function render() {
        label.textContent = d.label || "";
        tick();
      }

      render();
      const timer = setInterval(tick, 250);
      return {
        update(next) {
          d = next;
          render();
        },
        unmount() {
          clearInterval(timer);
        },
      };
    },
  });
})();

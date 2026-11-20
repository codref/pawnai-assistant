/* Radio plugin — "radio" widget: station card, now playing, EQ, volume. */
(function () {
  "use strict";

  const { el } = window.PawnWidgets;
  const BARS = 14;

  window.PawnWidgets.register("radio", {
    mount(body, data, api) {
      const root = el("div", "rd");

      const top = el("div", "rd-top");
      const logoWrap = el("div", "rd-logo");
      const img = el("img");
      img.alt = "";
      img.referrerPolicy = "no-referrer";
      img.decoding = "async";
      const glyph = el("span", "rd-glyph", "♫");
      logoWrap.append(img, glyph);
      img.addEventListener("load", () => logoWrap.classList.add("has-img"));
      img.addEventListener("error", () => logoWrap.classList.remove("has-img"));

      const id = el("div", "rd-id");
      const name = el("div", "rd-name");
      const slogan = el("div", "rd-slogan");
      const chips = el("div", "rd-chips");
      id.append(name, slogan, chips);
      top.append(logoWrap, id);

      const np = el("div", "rd-np");
      const npTag = el("span", "rd-np-tag", "NOW");
      const npView = el("div", "rd-np-view");
      const npText = el("span", "rd-np-text");
      npView.appendChild(npText);
      np.append(npTag, npView);

      const eq = el("div", "rd-eq");
      eq.setAttribute("aria-hidden", "true");
      for (let i = 0; i < BARS; i += 1) {
        const b = el("i");
        b.style.animationDelay = `${(-(i * 137) % 900) / 1000}s`;
        b.style.animationDuration = `${0.55 + ((i * 53) % 40) / 100}s`;
        eq.appendChild(b);
      }

      const ctl = el("div", "rd-ctl");
      const volLabel = el("label", "rd-vol");
      const volTxt = el("span", null, "VOL");
      const vol = el("input");
      vol.type = "range";
      vol.min = "0";
      vol.max = "100";
      vol.step = "1";
      vol.setAttribute("aria-label", "Radio volume");
      const volOut = el("output");
      volLabel.append(volTxt, vol, volOut);
      const badge = el("span", "rd-badge");
      const stop = el("button", "hud-btn danger", "STOP");
      stop.type = "button";
      stop.addEventListener("click", () => api.action("stop"));
      ctl.append(volLabel, badge, stop);

      const err = el("div", "rd-err");
      root.append(top, np, eq, ctl, err);
      body.appendChild(root);

      let dragging = false;
      let volTimer = null;
      vol.addEventListener("pointerdown", () => (dragging = true));
      vol.addEventListener("pointerup", () => (dragging = false));
      vol.addEventListener("input", () => {
        volOut.textContent = vol.value;
        clearTimeout(volTimer);
        volTimer = setTimeout(() => api.action("set_volume", { value: Number(vol.value) }), 120);
      });

      function chip(text) {
        return text ? el("span", "rd-chip", text) : null;
      }

      function render(d) {
        name.textContent = d.name || "—";
        slogan.textContent = d.slogan || "";
        if (d.logo && img.getAttribute("src") !== d.logo) img.src = d.logo;
        const freq = [d.band, d.frequency].filter(Boolean).join(" ");
        const rate = [d.bitrate ? `${d.bitrate}k` : "", d.format || ""].filter(Boolean).join(" ");
        chips.replaceChildren(
          ...[chip(d.genre), chip(d.location), chip(freq), chip(rate)].filter(Boolean)
        );
        const nowText = d.now_playing || "—";
        if (npText.textContent !== nowText) {
          npText.textContent = nowText;
          npView.classList.remove("scroll");
          requestAnimationFrame(() => {
            const over = npText.scrollWidth - npView.clientWidth;
            npView.classList.toggle("scroll", over > 4);
            npView.style.setProperty("--rd-shift", `${-over - 16}px`);
            npView.style.setProperty("--rd-dur", `${Math.max(6, (over + 40) / 30)}s`);
          });
        }
        if (!dragging && d.volume != null) {
          vol.value = String(d.volume);
          volOut.textContent = String(d.volume);
        }
        const status = d.status === "error" ? "error" : d.ducked ? "ducked" : "live";
        root.dataset.status = status;
        badge.textContent = status === "error" ? "OFFLINE" : status === "ducked" ? "DUCKED" : "● LIVE";
        err.textContent = d.status === "error" ? d.error || "stream lost" : "";
        api.setStatus(status === "live" ? (d.name || "").toUpperCase() : badge.textContent);
      }

      render(data);
      return {
        update: (d) => render(d),
        unmount() {
          clearTimeout(volTimer);
        },
      };
    },
  });
})();

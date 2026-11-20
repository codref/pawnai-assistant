/*
 * PawnWM — small window manager for the HUD stage.
 *
 *   PawnWM.init({ layer, tray })
 *   const win = PawnWM.open({ id, title, kind, anchor: "tr", w: 300, modal: false })
 *   win.body           // mount your content here
 *   win.setTitle(t); win.setSub(t); win.flash(); win.focus(); win.minimize(); win.restore()
 *   win.settle()       // after mounting content: re-place with the real size
 *   win.on("close", (reason) => …)   // reason: "user" | "server" | "replace" | "program"
 *   win.close(reason)
 *
 * Opening an id that is already open focuses and returns the existing window.
 * Positions/sizes are remembered per kind (localStorage, best effort).
 */
(function (global) {
  "use strict";

  const STORE = "pawn.wm.";
  const GAP = 14;
  const CASCADE = 26;
  const narrow = global.matchMedia ? global.matchMedia("(max-width: 900px)") : { matches: false };
  const reduceMotion = global.matchMedia
    ? global.matchMedia("(prefers-reduced-motion: reduce)")
    : { matches: false };

  let layer = null;
  let tray = null;
  let z = 10;
  let serial = 0;
  const wins = new Map();

  function load(kind) {
    try {
      const raw = global.localStorage.getItem(STORE + kind);
      return raw ? JSON.parse(raw) : null;
    } catch (_e) {
      return null;
    }
  }

  function save(kind, rect) {
    try {
      global.localStorage.setItem(STORE + kind, JSON.stringify(rect));
    } catch (_e) {
      /* private mode etc. */
    }
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function bounds() {
    return { w: layer.clientWidth, h: layer.clientHeight };
  }

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function topmost() {
    let best = null;
    for (const w of wins.values()) {
      if (w.minimized) continue;
      if (!best || w.z > best.z) best = w;
    }
    return best;
  }

  function placeAt(win, anchor, width, height) {
    const b = bounds();
    const same = [...wins.values()].filter((w) => w !== win && w.anchor === anchor && !w.minimized).length;
    const off = same * CASCADE;
    let x;
    let y;
    switch (anchor) {
      case "tl":
        x = GAP + off;
        y = GAP + off;
        break;
      case "bl":
        x = GAP + off;
        y = b.h - height - GAP - 40 - off;
        break;
      case "br":
        x = b.w - width - GAP - off;
        y = b.h - height - GAP - 40 - off;
        break;
      case "center":
        x = (b.w - width) / 2 + off;
        y = Math.max(GAP, (b.h - height) / 2.6) + off;
        break;
      default: // tr
        x = b.w - width - GAP - off;
        y = GAP + off;
    }
    x = clamp(x, 0, Math.max(0, b.w - width));
    y = clamp(y, 0, Math.max(0, b.h - height));
    // Don't land on top of another window: slide away from the anchor edge.
    const others = [...wins.values()].filter((w) => w !== win && !w.minimized);
    const down = anchor !== "bl" && anchor !== "br";
    for (let i = 0; i < 12; i += 1) {
      const hit = others.find((w) => {
        const r = { x: w.el.offsetLeft, y: w.el.offsetTop, w: w.el.offsetWidth, h: w.el.offsetHeight };
        return x < r.x + r.w && x + width > r.x && y < r.y + r.h && y + height > r.y;
      });
      if (!hit) break;
      const ny = down ? hit.el.offsetTop + hit.el.offsetHeight + GAP / 2 : hit.el.offsetTop - height - GAP / 2;
      if (ny < 0 || ny + height > b.h) break; // no room: accept the overlap (focus decides)
      y = ny;
    }
    return { x, y };
  }

  function setRect(win, r) {
    const b = bounds();
    if (r.w) win.el.style.width = `${Math.min(r.w, b.w)}px`;
    if (r.h) win.el.style.height = `${Math.min(r.h, b.h)}px`;
    const w = win.el.offsetWidth || r.w || 280;
    const h = win.el.offsetHeight || r.h || 120;
    win.el.style.left = `${clamp(r.x, 0, Math.max(0, b.w - w))}px`;
    win.el.style.top = `${clamp(r.y, 0, Math.max(0, b.h - h))}px`;
  }

  function rectOf(win) {
    return {
      x: win.el.offsetLeft,
      y: win.el.offsetTop,
      w: win.resized ? win.el.offsetWidth : undefined,
      h: win.resized ? win.el.offsetHeight : undefined,
    };
  }

  function dragger(win, handle, onMove) {
    handle.addEventListener("pointerdown", (ev) => {
      if (narrow.matches || ev.button !== 0) return;
      if (ev.target.closest("button, input, a, select, textarea")) return;
      ev.preventDefault();
      win.focus();
      handle.setPointerCapture(ev.pointerId);
      const start = { x: ev.clientX, y: ev.clientY, l: win.el.offsetLeft, t: win.el.offsetTop, w: win.el.offsetWidth, h: win.el.offsetHeight };
      const move = (e) => onMove(e.clientX - start.x, e.clientY - start.y, start);
      const up = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        handle.removeEventListener("pointercancel", up);
        win.pinned = true;
        save(win.kind, rectOf(win));
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
      handle.addEventListener("pointercancel", up);
    });
  }

  class Win {
    constructor(opts) {
      this.id = opts.id;
      this.kind = opts.kind || opts.id;
      this.anchor = opts.anchor || "tr";
      this.modal = !!opts.modal;
      this.closable = opts.closable !== false;
      this.minimizable = opts.minimizable !== false && !this.modal;
      this.minimized = false;
      this.resized = false;
      this.closed = false;
      this.handlers = { close: [], focus: [] };
      this.idx = String((++serial) % 100).padStart(2, "0");
      this.z = ++z;

      const root = el("section", "wm-win opening");
      root.setAttribute("role", "dialog");
      const titleId = `wm-t-${serial}`;
      root.setAttribute("aria-labelledby", titleId);
      if (this.modal) root.setAttribute("aria-modal", "true");
      root.dataset.kind = this.kind;
      root.tabIndex = -1;

      const frame = el("div", "wm-frame");
      const inner = el("div", "wm-inner");
      const head = el("header", "wm-head");
      head.appendChild(el("span", "wm-idx", `[${this.idx}]`));
      this.titleEl = el("h3", "wm-title", opts.title || this.kind.toUpperCase());
      this.titleEl.id = titleId;
      head.appendChild(this.titleEl);
      this.subEl = el("span", "wm-sub", "");
      head.appendChild(this.subEl);
      const ctl = el("div", "wm-ctl");
      if (this.minimizable) {
        const min = el("button", "wm-btn wm-min", "▁");
        min.type = "button";
        min.setAttribute("aria-label", "Minimize");
        min.addEventListener("click", () => this.minimize());
        ctl.appendChild(min);
      }
      if (this.closable) {
        const x = el("button", "wm-btn wm-x", "✕");
        x.type = "button";
        x.setAttribute("aria-label", "Close");
        x.addEventListener("click", () => this.close("user"));
        ctl.appendChild(x);
      }
      head.appendChild(ctl);
      this.body = el("div", "wm-body");
      inner.append(head, this.body);
      frame.appendChild(inner);
      root.append(frame, el("i", "wm-corner tl"), el("i", "wm-corner br"));
      if (opts.resizable !== false) {
        const grip = el("div", "wm-grip");
        grip.setAttribute("aria-hidden", "true");
        root.appendChild(grip);
        dragger(this, grip, (dx, dy, s) => {
          const b = bounds();
          this.resized = true;
          this.el.style.width = `${clamp(s.w + dx, 220, b.w - this.el.offsetLeft)}px`;
          this.el.style.height = `${clamp(s.h + dy, 96, b.h - this.el.offsetTop)}px`;
        });
      }
      this.el = root;
      this.head = head;

      dragger(this, head, (dx, dy, s) => {
        const b = bounds();
        this.el.style.left = `${clamp(s.l + dx, 0, b.w - this.el.offsetWidth)}px`;
        this.el.style.top = `${clamp(s.t + dy, 0, Math.max(0, b.h - this.el.offsetHeight))}px`;
      });
      root.addEventListener("pointerdown", () => this.focus(), true);
      root.addEventListener("animationend", (e) => {
        if (e.target === root) root.classList.remove("opening", "flash");
      });

      if (this.modal) {
        this.backdrop = el("div", "wm-backdrop");
        this.backdrop.addEventListener("click", () => this.closable && this.close("user"));
        layer.appendChild(this.backdrop);
      }
      layer.appendChild(root);

      const remembered = load(this.kind);
      this.pinned = !!(remembered && remembered.x != null && !this.modal);
      const w = (remembered && remembered.w) || opts.w || 300;
      root.style.width = `${w}px`;
      if (remembered && remembered.h) {
        root.style.height = `${remembered.h}px`;
        this.resized = true;
      } else if (opts.h) {
        root.style.height = `${opts.h}px`;
      }
      const pos =
        remembered && !this.modal && remembered.x != null
          ? { x: remembered.x, y: remembered.y }
          : placeAt(this, this.anchor, root.offsetWidth || w, root.offsetHeight || 160);
      setRect(this, { x: pos.x, y: pos.y });
      if (reduceMotion.matches) root.classList.remove("opening");
      this.focus();
    }

    on(evt, fn) {
      (this.handlers[evt] || (this.handlers[evt] = [])).push(fn);
      return this;
    }

    emit(evt, arg) {
      for (const fn of this.handlers[evt] || []) {
        try {
          fn(arg);
        } catch (e) {
          console.error(e);
        }
      }
    }

    /** Re-place after content was mounted (real height known), unless the user placed it. */
    settle() {
      if (this.pinned || this.closed) return;
      const pos = placeAt(this, this.anchor, this.el.offsetWidth, this.el.offsetHeight);
      setRect(this, pos);
    }

    setTitle(text) {
      this.titleEl.textContent = text;
      if (this.chip) this.chip.querySelector(".ttl").textContent = text;
    }

    setSub(text) {
      if (this.subEl.textContent !== text) this.subEl.textContent = text;
      if (this.chip) this.chip.querySelector(".sub").textContent = text;
    }

    flash() {
      this.el.classList.remove("flash");
      void this.el.offsetWidth; // restart the animation
      this.el.classList.add("flash");
      if (this.chip) this.chip.classList.add("alert");
    }

    focus() {
      if (this.closed) return;
      if (this.minimized) this.restore();
      for (const w of wins.values()) w.el.classList.toggle("focused", w === this);
      if (this.z < z) this.z = ++z;
      if (this.backdrop) this.backdrop.style.zIndex = String(this.z);
      this.el.style.zIndex = String(this.z);
      this.emit("focus");
    }

    minimize() {
      if (!this.minimizable || this.minimized) return;
      this.minimized = true;
      this.el.hidden = true;
      this.el.classList.remove("focused");
      const chip = el("button", "wm-chip");
      chip.type = "button";
      chip.append(el("span", "idx", `[${this.idx}]`), el("span", "ttl", this.titleEl.textContent), el("span", "sub", this.subEl.textContent));
      chip.setAttribute("aria-label", `Restore ${this.titleEl.textContent}`);
      chip.addEventListener("click", () => this.restore());
      tray.appendChild(chip);
      this.chip = chip;
      const next = topmost();
      if (next) next.focus();
    }

    restore() {
      if (!this.minimized) return;
      this.minimized = false;
      this.el.hidden = false;
      if (this.chip) this.chip.remove();
      this.chip = null;
      this.focus();
    }

    close(reason) {
      if (this.closed) return;
      this.closed = true;
      wins.delete(this.id);
      if (this.chip) this.chip.remove();
      if (this.backdrop) this.backdrop.remove();
      this.emit("close", reason || "program");
      const root = this.el;
      if (reduceMotion.matches) {
        root.remove();
      } else {
        root.classList.add("closing");
        setTimeout(() => root.remove(), 220);
      }
      const next = topmost();
      if (next) next.focus();
    }
  }

  const PawnWM = {
    init({ layer: l, tray: t }) {
      layer = l;
      tray = t;
      document.addEventListener("keydown", (ev) => {
        if (ev.key !== "Escape") return;
        const w = topmost();
        if (w && w.closable) {
          ev.preventDefault();
          w.close("user");
        }
      });
      global.addEventListener("resize", () => {
        for (const w of wins.values()) setRect(w, { x: w.el.offsetLeft, y: w.el.offsetTop });
      });
    },

    open(opts) {
      if (!layer) throw new Error("PawnWM.init first");
      const existing = wins.get(opts.id);
      if (existing) {
        if (opts.title) existing.setTitle(opts.title);
        existing.focus();
        return existing;
      }
      const win = new Win(opts);
      wins.set(win.id, win);
      return win;
    },

    get(id) {
      return wins.get(id) || null;
    },

    close(id, reason) {
      const w = wins.get(id);
      if (w) w.close(reason || "program");
    },

    list() {
      return [...wins.values()];
    },
  };

  global.PawnWM = PawnWM;
})(window);

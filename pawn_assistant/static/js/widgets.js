/*
 * PawnWidgets — widget renderers for command windows + the server UI host.
 *
 * A command plugin ships a widget file (listed in hello.assets) that calls:
 *
 *   PawnWidgets.register("countdown", {
 *     mount(body, data, api) {          // api: action(name, data), close(), setStatus(text),
 *       …                               //      flash(), now() (server-clock ms), win
 *       return { update(data, changed) {…}, unmount() {…} };
 *     },
 *   });
 *
 * The host turns {"type":"ui", op: open|update|close} messages into PawnWM
 * windows; closing a window by hand sends {"type":"ui_action", action:"close"}.
 */
(function (global) {
  "use strict";

  const kinds = new Map();
  const loaded = new Set();

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function label(key) {
    return String(key).replace(/_/g, " ");
  }

  // ---------------------------------------------------------------- built-in kinds

  kinds.set("kv", {
    mount(body, data) {
      const dl = el("dl", "wg-kv");
      body.appendChild(dl);
      const render = (d) => {
        dl.replaceChildren();
        for (const [k, v] of Object.entries(d)) {
          if (v == null || v === "" || typeof v === "object") continue;
          dl.append(el("dt", null, label(k)), el("dd", null, String(v)));
        }
      };
      render(data);
      return { update: (d) => render(d), unmount() {} };
    },
  });

  kinds.set("text", {
    mount(body, data) {
      const p = el("p", "wg-text", data.text || "");
      body.appendChild(p);
      return { update: (d) => (p.textContent = d.text || ""), unmount() {} };
    },
  });

  // ---------------------------------------------------------------- asset loader

  function loadOne(url) {
    if (loaded.has(url)) return Promise.resolve();
    loaded.add(url);
    return new Promise((resolve) => {
      let node;
      if (url.endsWith(".css")) {
        node = document.createElement("link");
        node.rel = "stylesheet";
        node.href = url;
      } else {
        node = document.createElement("script");
        node.src = url;
        node.async = false; // keep declaration order
      }
      node.onload = () => resolve();
      node.onerror = () => {
        console.warn("widget asset failed:", url);
        resolve();
      };
      document.head.appendChild(node);
    });
  }

  // ---------------------------------------------------------------- host

  function createHost({ send, skew }) {
    const open = new Map(); // id → { win, inst, data }
    let queue = Promise.resolve();

    const now = () => Date.now() + skew();

    function mount(msg) {
      const win = global.PawnWM.open({
        id: msg.id,
        kind: msg.kind,
        title: msg.title || (msg.kind || "").toUpperCase(),
        anchor: msg.anchor || "tr",
        w: msg.w || 300,
      });
      const entry = { win, inst: null, data: { ...(msg.data || {}) } };
      const api = {
        win,
        now,
        action(name, data) {
          send({ type: "ui_action", id: msg.id, action: name, data: data || {} });
        },
        close() {
          win.close("user");
        },
        setStatus(text) {
          win.setSub(text || "");
        },
        flash() {
          win.flash();
        },
      };
      const def = kinds.get(msg.kind) || kinds.get("kv");
      try {
        entry.inst = def.mount(win.body, entry.data, api) || null;
      } catch (e) {
        console.error("widget mount failed", msg.kind, e);
      }
      win.settle();
      win.on("close", (reason) => {
        open.delete(msg.id);
        if (entry.inst && entry.inst.unmount) {
          try {
            entry.inst.unmount();
          } catch (e) {
            console.error(e);
          }
        }
        if (reason === "user") send({ type: "ui_action", id: msg.id, action: "close", data: {} });
      });
      open.set(msg.id, entry);
      return entry;
    }

    function apply(msg) {
      const entry = open.get(msg.id);
      if (msg.op === "open") {
        if (entry) {
          entry.data = { ...(msg.data || {}) };
          if (msg.title) entry.win.setTitle(msg.title);
          if (entry.inst && entry.inst.update) entry.inst.update(entry.data, entry.data);
        } else {
          mount(msg);
        }
      } else if (msg.op === "update" && entry) {
        Object.assign(entry.data, msg.data || {});
        if (msg.title) entry.win.setTitle(msg.title);
        if (entry.inst && entry.inst.update) entry.inst.update(entry.data, msg.data || {});
      } else if (msg.op === "close" && entry) {
        entry.win.close("server");
      }
    }

    return {
      /** Load plugin assets; messages handled meanwhile wait for them. */
      load(assets) {
        const urls = [...((assets && assets.css) || []), ...((assets && assets.js) || [])];
        queue = queue.then(() => Promise.all(urls.map(loadOne)));
        return queue;
      },
      handle(msg) {
        queue = queue.then(() => apply(msg)).catch((e) => console.error(e));
      },
      /** hello.ui: reopen the server's windows, close ones that are gone. */
      replay(list) {
        queue = queue.then(() => {
          const ids = new Set((list || []).map((m) => m.id));
          for (const [id, entry] of open) if (!ids.has(id)) entry.win.close("server");
          for (const m of list || []) apply({ ...m, op: "open" });
        });
      },
    };
  }

  global.PawnWidgets = {
    register(kind, def) {
      kinds.set(kind, def);
    },
    has(kind) {
      return kinds.has(kind);
    },
    createHost,
    el,
  };
})(window);

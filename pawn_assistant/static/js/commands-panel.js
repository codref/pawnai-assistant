/*
 * MODULES window — enable/disable command plugins.
 *
 *   PawnModules.init({ button, send })
 *   PawnModules.setCommands(list)   // hello.commands / {"type":"commands"}
 */
(function (global) {
  "use strict";

  const { el } = global.PawnWidgets;
  const WIN_ID = "sys:modules";
  let commands = [];
  let send = () => {};
  let win = null;
  const pending = new Set();

  function render() {
    if (!win) return;
    const body = win.body;
    body.replaceChildren();

    const enabled = commands.filter((c) => c.enabled).length;
    win.setSub(`${enabled}/${commands.length} online`);

    const intro = el("p", "mod-intro", "Command modules loaded into PAWN. Disabled modules drop out of the fast-path and the intent model.");
    body.appendChild(intro);

    const list = el("ul", "mod-list");
    list.setAttribute("role", "list");
    for (const c of commands) {
      const li = el("li", "mod-item");
      li.dataset.enabled = String(c.enabled);
      li.dataset.active = String(!!c.active);
      if (pending.has(c.name)) li.classList.add("pending");

      const led = el("span", "mod-led");
      led.setAttribute("aria-hidden", "true");

      const head = el("div", "mod-head");
      const title = el("h4", "mod-title", c.title);
      const tag = el("span", "mod-tag", c.core ? "core" : c.name);
      head.append(led, title, tag);

      const sw = el("label", "mod-switch");
      const input = el("input");
      input.type = "checkbox";
      input.setAttribute("role", "switch");
      input.checked = !!c.enabled;
      input.disabled = !!c.core || pending.has(c.name);
      input.setAttribute("aria-label", `${c.title} ${c.enabled ? "enabled" : "disabled"}`);
      input.addEventListener("change", () => {
        pending.add(c.name);
        send({ type: "set_command", name: c.name, enabled: input.checked });
        render();
      });
      const track = el("span", "mod-track");
      const state = el("span", "mod-state", c.core ? "LOCKED" : c.enabled ? "ON" : "OFF");
      sw.append(input, track, state);
      head.appendChild(sw);

      const desc = el("p", "mod-desc", c.description || "");
      const intents = el("ul", "mod-intents");
      for (const i of c.intents || []) {
        const chip = el("li", "mod-intent");
        chip.append(el("span", "n", i.name));
        if (i.slots && i.slots.length) chip.append(el("span", "s", `{${i.slots.join(", ")}}`));
        if (i.example) chip.append(el("span", "x", `“${i.example}”`));
        chip.title = i.description || "";
        intents.appendChild(chip);
      }
      li.append(head, desc, intents);
      list.appendChild(li);
    }
    body.appendChild(list);
  }

  function open() {
    win = global.PawnWM.open({
      id: WIN_ID,
      kind: "modules",
      title: "MODULES",
      anchor: "center",
      modal: true,
      w: 560,
    });
    win.on("close", () => {
      win = null;
    });
    render();
    win.settle();
    const first = win.body.querySelector("input:not([disabled])");
    if (first) first.focus({ preventScroll: true });
  }

  function toggle() {
    if (win) win.close("user");
    else open();
  }

  global.PawnModules = {
    init({ button, send: s }) {
      send = s;
      button.addEventListener("click", toggle);
      document.addEventListener("keydown", (ev) => {
        if (ev.key !== "m" && ev.key !== "M") return;
        if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
        const t = ev.target;
        if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
        toggle();
      });
    },
    setCommands(list) {
      commands = list || [];
      pending.clear();
      render();
    },
    open,
  };
})(window);

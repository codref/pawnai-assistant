/* global PawnPresence, PawnWM, PawnWidgets, PawnModules */
(function () {
  "use strict";

  const els = {
    chip: document.getElementById("state-chip"),
    device: document.getElementById("d-device"),
    rate: document.getElementById("d-rate"),
    overruns: document.getElementById("d-overruns"),
    rms: document.getElementById("d-rms"),
    peak: document.getElementById("d-peak"),
    barRms: document.getElementById("bar-rms"),
    wakeModel: document.getElementById("d-wake-model"),
    wakeScore: document.getElementById("d-wake-score"),
    wakeThr: document.getElementById("d-wake-thr"),
    vad: document.getElementById("d-vad"),
    silence: document.getElementById("d-silence"),
    dwell: document.getElementById("d-dwell"),
    whisper: document.getElementById("d-whisper"),
    lastDur: document.getElementById("d-last-dur"),
    infer: document.getElementById("d-infer"),
    rtf: document.getElementById("d-rtf"),
    rss: document.getElementById("d-rss"),
    intent: document.getElementById("d-intent"),
    intentSrc: document.getElementById("d-intent-src"),
    parse: document.getElementById("d-parse"),
    ttsMs: document.getElementById("d-tts-ms"),
    llm: document.getElementById("d-llm"),
    voice: document.getElementById("d-voice"),
    log: document.getElementById("transcript-log"),
    spark: document.getElementById("wake-spark"),
    gain: document.getElementById("gain"),
    gainLabel: document.getElementById("gain-label"),
    gainOut: document.getElementById("gain-out"),
    gainKnob: document.getElementById("gain-knob"),
  };
  let ws = null;
  let gainDragging = false;
  let gainTimer = null;
  let gainUnit = "%";

  const presence = PawnPresence.mountPresence(document.getElementById("presence"));

  // Command windows: plugins open/update/close them via {"type":"ui"} messages.
  let clockSkew = 0; // server clock − local clock (ms), from hello.now_ms
  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }
  PawnWM.init({
    layer: document.getElementById("wm-layer"),
    tray: document.getElementById("wm-tray"),
  });
  const uiHost = PawnWidgets.createHost({ send, skew: () => clockSkew });
  PawnModules.init({ button: document.getElementById("modules-btn"), send });
  const sparkScores = [];
  const SPARK_N = 80;
  let wakeThreshold = 0.5;
  let reconnectDelay = 500;

  function setText(el, value) {
    if (el.textContent !== value) el.textContent = value;
  }

  function setState(state) {
    if (els.chip.dataset.state === state) return;
    els.chip.textContent = state;
    els.chip.dataset.state = state;
    presence.setTelemetry({ state });
  }

  function dbToPct(db) {
    const pct = ((db + 60) / 50) * 100;
    return Math.max(0, Math.min(100, pct));
  }

  function drawSpark() {
    const c = els.spark;
    const ctx = c.getContext("2d");
    const w = c.width;
    const h = c.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#1a6a78";
    ctx.beginPath();
    const thrY = h - wakeThreshold * h;
    ctx.moveTo(0, thrY);
    ctx.lineTo(w, thrY);
    ctx.stroke();
    if (sparkScores.length < 2) return;
    ctx.strokeStyle = "#3de7ff";
    ctx.beginPath();
    sparkScores.forEach((s, i) => {
      const x = (i / (SPARK_N - 1)) * w;
      const y = h - Math.max(0, Math.min(1, s)) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  function pushSpark(score) {
    sparkScores.push(score);
    while (sparkScores.length > SPARK_N) sparkScores.shift();
    drawSpark();
  }

  function addTranscript(ev) {
    const li = document.createElement("li");
    if (!ev.text) {
      li.className = "empty";
      li.textContent = ev.reason
        ? `(empty utterance — ${ev.reason})`
        : "(empty utterance)";
    } else {
      li.textContent = ev.text;
    }
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `${ev.duration_ms} ms audio · ${ev.infer_ms} ms infer · rtf ${ev.rtf}`;
    li.appendChild(meta);
    els.log.prepend(li);
    while (els.log.children.length > 40) els.log.removeChild(els.log.lastChild);
  }

  function appendToLatest(cls, text) {
    const li = els.log.firstElementChild;
    if (!li) return;
    const span = document.createElement("span");
    span.className = cls;
    span.textContent = text;
    li.appendChild(span);
  }

  function renderGain(g) {
    els.gain.hidden = false;
    setText(els.gainLabel, g.label);
    els.gainKnob.min = g.min;
    els.gainKnob.max = g.max;
    els.gainKnob.step = g.step;
    gainUnit = g.unit;
    if (!gainDragging) els.gainKnob.value = g.value;
    showGainValue(g.value, g.db);
  }

  function showGainValue(value, db) {
    const v = Number(value);
    const base = gainUnit === "dB" ? `${v > 0 ? "+" : ""}${v.toFixed(1)} dB` : `${Math.round(v)}%`;
    const extra = gainUnit !== "dB" && db != null ? ` · ${db > 0 ? "+" : ""}${Number(db).toFixed(1)} dB` : "";
    setText(els.gainOut, base + extra);
  }

  els.gainKnob.addEventListener("pointerdown", () => { gainDragging = true; });
  els.gainKnob.addEventListener("pointerup", () => { gainDragging = false; });
  els.gainKnob.addEventListener("input", () => {
    showGainValue(els.gainKnob.value, null);
    // Throttle: amixer is a subprocess, one call per 120 ms is plenty.
    clearTimeout(gainTimer);
    gainTimer = setTimeout(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "set_gain", value: Number(els.gainKnob.value) }));
      }
    }, 120);
  });

  function onMeter(m) {
    setText(els.device, m.device_name || "—");
    setText(els.rate, `${m.sample_rate} Hz`);
    setText(els.overruns, String(m.overruns ?? 0));
    setText(els.rms, `${m.rms_db.toFixed(1)} dBFS`);
    setText(els.peak, `${m.peak_db.toFixed(1)} dBFS`);
    els.peak.classList.toggle("hot", m.peak_db >= -1.0);
    const barW = `${dbToPct(m.rms_db).toFixed(1)}%`;
    if (els.barRms.style.width !== barW) els.barRms.style.width = barW;
    setText(els.wakeScore, m.wake_score.toFixed(3));
    setText(els.wakeThr, m.wake_threshold.toFixed(3));
    wakeThreshold = m.wake_threshold;
    setText(els.vad, m.vad_prob.toFixed(3));
    setText(els.silence, `${m.silence_ms} ms`);
    setText(els.dwell, `${m.since_ms ?? 0} ms`);
    setText(els.whisper, m.whisper_resident ? "resident" : "cold");
    setText(els.rss, `${m.rss_mb} MB`);
    setText(els.llm, m.llm_resident ? "resident" : "cold");
    setText(els.voice, m.tts_resident ? "resident" : "cold");
    setState(m.state);
    pushSpark(m.wake_score);
    presence.setTelemetry({
      state: m.state,
      wake_score: m.wake_score,
      rms_db: m.rms_db,
    });
  }

  function onEvent(ev) {
    if (ev.name === "state") {
      setState(ev.state);
    } else if (ev.name === "beep") {
      presence.setTelemetry({ beep: true, state: "AWAKE" });
    } else if (ev.name === "transcript") {
      addTranscript(ev);
      els.lastDur.textContent = `${ev.duration_ms} ms`;
      els.infer.textContent = `${ev.infer_ms} ms`;
      els.rtf.textContent = String(ev.rtf);
    } else if (ev.name === "intent") {
      const slots = Object.entries(ev.slots || {})
        .map(([k, v]) => `${k}=${v}`)
        .join(" ");
      setText(els.intent, slots ? `${ev.intent} ${slots}` : ev.intent);
      setText(els.intentSrc, ev.source);
      setText(els.parse, `${ev.parse_ms} ms`);
      appendToLatest("intent", `→ ${ev.intent}${slots ? " " + slots : ""} · ${ev.source} · ${ev.parse_ms} ms`);
    } else if (ev.name === "reply") {
      setText(els.ttsMs, `${ev.tts_ms} ms`);
      // Phrased replies stream: seq > 0 updates the line opened by seq 0.
      const cont = (ev.seq || 0) > 0;
      if (ev.kind === "alert") {
        const first = els.log.firstElementChild;
        if (cont && first && first.classList.contains("alert")) {
          first.textContent = `⏰ ${ev.text}`;
        } else {
          const li = document.createElement("li");
          li.className = "alert";
          li.textContent = `⏰ ${ev.text}`;
          els.log.prepend(li);
        }
      } else {
        const latest = els.log.firstElementChild;
        const spans = latest ? latest.querySelectorAll("span.reply") : [];
        if (cont && spans.length) {
          spans[spans.length - 1].textContent = `PAWN: ${ev.text}`;
        } else {
          appendToLatest("reply", `PAWN: ${ev.text}`);
        }
      }
    } else if (ev.name === "fault") {
      setState("FAULT");
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = `FAULT: ${ev.message}`;
      els.log.prepend(li);
    }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => {
      reconnectDelay = 500;
    };

    ws.onmessage = (msg) => {
      let data;
      try {
        data = JSON.parse(msg.data);
      } catch {
        return;
      }
      if (data.type === "hello") {
        els.wakeModel.textContent = data.wake_model;
        els.wakeThr.textContent = Number(data.wake_threshold).toFixed(3);
        els.device.textContent = data.device_name;
        els.rate.textContent = `${data.sample_rate} Hz`;
        if (data.status && data.status.state) setState(data.status.state);
        if (data.gain) renderGain(data.gain);
        if (typeof data.now_ms === "number") clockSkew = data.now_ms - Date.now();
        uiHost.load(data.assets);
        uiHost.replay(data.ui || []);
        PawnModules.setCommands(data.commands || []);
        return;
      }
      if (data.type === "ui") {
        uiHost.handle(data);
        return;
      }
      if (data.type === "commands") {
        PawnModules.setCommands(data.commands);
        return;
      }
      if (data.type === "gain") {
        renderGain(data);
        return;
      }
      if (data.type === "meter") onMeter(data);
      else if (data.type === "event") onEvent(data);
    };

    ws.onclose = () => {
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(5000, reconnectDelay * 1.5);
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  connect();
})();

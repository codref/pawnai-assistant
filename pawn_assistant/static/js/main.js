/* global PawnPresence */
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
    log: document.getElementById("transcript-log"),
    spark: document.getElementById("wake-spark"),
  };

  const presence = PawnPresence.mountPresence(document.getElementById("presence"));
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

  function onMeter(m) {
    setText(els.device, m.device_name || "—");
    setText(els.rate, `${m.sample_rate} Hz`);
    setText(els.overruns, String(m.overruns ?? 0));
    setText(els.rms, `${m.rms_db.toFixed(1)} dBFS`);
    setText(els.peak, `${m.peak_db.toFixed(1)} dBFS`);
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
    const ws = new WebSocket(`${proto}://${location.host}/ws`);

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

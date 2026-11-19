/* global Phaser */
(function (global) {
  "use strict";

  const COLORS = {
    IDLE: 0x1a6a78,
    STIR: 0xffb020,
    AWAKE: 0x3de7ff,
    LISTEN: 0x3de7ff,
    HOLD: 0x7ad4e0,
    BUSY: 0xffb020,
    THINK: 0xc77dff,
    SPEAK: 0x7dffb0,
    FAULT: 0xff3d6e,
  };

  class PresenceScene extends Phaser.Scene {
    constructor() {
      super("presence");
      this.pawnState = "IDLE";
      this.wakeScore = 0;
      this.rmsDb = -60;
      this.flash = 0;
      this.glyphColor = null;
    }

    create() {
      const { width, height } = this.scale;
      this.cameras.main.setBackgroundColor("#05080c");

      this.ring = this.add.graphics();
      this.core = this.add.graphics();
      this.glyph = this.add.text(width / 2, height / 2, "♟", {
        fontFamily: "Orbitron, sans-serif",
        fontSize: "72px",
        color: "#3de7ff",
      });
      this.glyph.setOrigin(0.5);

      this.tweens.add({
        targets: this.glyph,
        scale: { from: 0.96, to: 1.04 },
        duration: 2200,
        yoyo: true,
        repeat: -1,
        ease: "Sine.easeInOut",
      });

      this.scale.on("resize", (gameSize) => {
        const w = gameSize.width;
        const h = gameSize.height;
        this.glyph.setPosition(w / 2, h / 2);
      });
    }

    setTelemetry(partial) {
      if (partial.state) this.pawnState = partial.state;
      if (typeof partial.wake_score === "number") this.wakeScore = partial.wake_score;
      if (typeof partial.rms_db === "number") this.rmsDb = partial.rms_db;
      if (partial.beep) this.flash = 1;
    }

    update(_t, dt) {
      const { width, height } = this.scale;
      const cx = width / 2;
      const cy = height / 2;
      const color = COLORS[this.pawnState] || COLORS.IDLE;
      const level = Phaser.Math.Clamp((this.rmsDb + 60) / 50, 0, 1);
      const stir = Phaser.Math.Clamp(this.wakeScore, 0, 1);

      if (this.flash > 0) {
        this.flash = Math.max(0, this.flash - dt / 280);
      }

      const baseR = Math.min(width, height) * 0.18;
      const pulse =
        this.pawnState === "IDLE"
          ? 1 + 0.03 * Math.sin(this.time.now / 700)
          : 1 + 0.08 * stir + 0.12 * level + 0.2 * this.flash;

      this.ring.clear();
      this.ring.lineStyle(2, color, 0.35 + 0.45 * stir);
      this.ring.strokeCircle(cx, cy, baseR * pulse);
      this.ring.lineStyle(1, color, 0.2);
      this.ring.strokeCircle(cx, cy, baseR * pulse * (1.25 + 0.15 * level));

      if (this.pawnState === "LISTEN" || this.pawnState === "HOLD") {
        this.ring.lineStyle(3, 0x3de7ff, 0.55);
        this.ring.strokeCircle(cx, cy, baseR * (1.05 + 0.35 * level));
      }

      this.core.clear();
      const alpha = this.pawnState === "FAULT" ? 0.85 : 0.25 + 0.5 * stir + 0.3 * this.flash;
      this.core.fillStyle(color, alpha);
      this.core.fillCircle(cx, cy, baseR * 0.42 * pulse);

      // Text.setColor re-rasterizes the glyph canvas and re-uploads the texture,
      // so only call it when the colour actually changes (not every frame).
      if (color !== this.glyphColor) {
        this.glyphColor = color;
        this.glyph.setColor("#" + color.toString(16).padStart(6, "0"));
      }
      this.glyph.setAlpha(0.55 + 0.45 * Math.max(stir, this.flash, this.pawnState === "BUSY" || this.pawnState === "THINK" ? 0.7 : 0));

      if (this.pawnState === "SPEAK") {
        const wave = 0.5 + 0.5 * Math.sin(this.time.now / 110);
        this.ring.lineStyle(3, 0x7dffb0, 0.35 + 0.4 * wave);
        this.ring.strokeCircle(cx, cy, baseR * (1.08 + 0.12 * wave));
      }

      if (this.pawnState === "BUSY" || this.pawnState === "THINK") {
        this.glyph.rotation = (this.time.now / 900) % (Math.PI * 2);
      } else {
        this.glyph.rotation *= 0.9;
      }
    }
  }

  function mountPresence(parent) {
    const game = new Phaser.Game({
      type: Phaser.AUTO,
      parent,
      backgroundColor: "#05080c",
      scale: {
        mode: Phaser.Scale.RESIZE,
        autoCenter: Phaser.Scale.CENTER_BOTH,
        width: parent.clientWidth || 640,
        height: parent.clientHeight || 480,
      },
      scene: [PresenceScene],
      banner: false,
      // Ambient HUD: 30 fps is plenty and halves (or more, on 120/144 Hz
      // displays) the per-frame redraw cost versus the uncapped rAF loop.
      fps: { target: 30, limit: 30 },
      render: { powerPreference: "low-power" },
    });

    return {
      setTelemetry(partial) {
        const scene = game.scene.getScene("presence");
        if (scene && scene.setTelemetry) scene.setTelemetry(partial);
      },
      destroy() {
        game.destroy(true);
      },
    };
  }

  global.PawnPresence = { mountPresence };
})(window);

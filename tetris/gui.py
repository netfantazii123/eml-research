"""
gui.py — GUI с двумя вкладками: обучение оракула (PPO) и EML-дистилляция.

Запуск: python gui.py  или  python main.py gui

Вкладка "Training":
  Left   — редактор гиперпараметров PPO (Apply применяет на лету во время обучения)
  Center — графики avg/max линий и потерь (pg/vf/entropy) + лог
  Right  — числовой статус в реальном времени
  Bottom — Start / Stop / Save Model

Вкладка "Distillation":
  Left   — параметры дистилляции (датасет, DATA/GAME-фазы)
  Center — графики DATA-fitness (6 действий) и JOINT-score + лог
  Right  — статус (фаза, формулы, EML vs Oracle)
  Bottom — Start / Stop / Save Formulas
"""

import os
import glob
import json
import time
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
import storage
import reports
import runs as runs_mod
import pieces
from env import TetrisEnv
from cnn_oracle import TetrisCNN, get_device, describe_device, obs_to_tensors
from eml_distiller import EMLNode, EMLPolicy, play_episodes

# ── Catppuccin Mocha palette ─────────────────────────────────────────────────
BG     = "#1e1e2e"
BG2    = "#2a2a3e"
BG3    = "#313145"
FG     = "#cdd6f4"
FG2    = "#a6adc8"
ACCENT = "#89b4fa"
GREEN  = "#a6e3a1"
RED    = "#f38ba8"
YELLOW = "#f9e2af"
PURPLE = "#cba6f7"


# ── Widget helpers ────────────────────────────────────────────────────────────

def _sep(parent: tk.Widget) -> None:
    tk.Frame(parent, bg=BG3, height=1).pack(fill=tk.X, padx=4, pady=3)


def _btn(parent: tk.Widget, text: str, cmd, *,
         bg: str = BG3, fg: str = FG, state=tk.NORMAL) -> tk.Button:
    return tk.Button(
        parent, text=text, command=cmd,
        bg=bg, fg=fg, activebackground=fg, activeforeground=bg,
        font=("Consolas", 9, "bold"), relief=tk.FLAT, bd=0,
        padx=12, pady=6, state=state, cursor="hand2",
    )


class ParamRow:
    """Строка «Label + Entry» для одного гиперпараметра."""

    def __init__(self, parent: tk.Widget, label: str, default, fmt=float):
        self._fmt = fmt
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(row, text=label, bg=BG2, fg=FG2,
                 font=("Consolas", 8), width=13, anchor="w").pack(side=tk.LEFT)
        self._var = tk.StringVar(value=str(default))
        tk.Entry(row, textvariable=self._var,
                 bg=BG3, fg=FG, insertbackground=FG,
                 font=("Consolas", 8), width=11,
                 relief=tk.FLAT, bd=2).pack(side=tk.LEFT, padx=(2, 0))

    def get(self):
        try:
            return self._fmt(self._var.get())
        except ValueError:
            return None

    def set(self, value) -> None:
        self._var.set(str(value))


# ── Main window ───────────────────────────────────────────────────────────────

class TrainingGUI:
    _CHART_MAX = 300      # точек истории на графике
    _POLL_MS   = 250      # интервал опроса очереди, мс

    def __init__(self, root: tk.Tk, container: tk.Widget):
        self.root = root
        self.container = container

        self._q: queue.Queue       = queue.Queue()
        self._stop_evt             = threading.Event()
        self._thread: threading.Thread | None = None
        self._model: TetrisCNN | None         = None
        self._overrides: dict      = {}   # shared with training thread (GIL-safe reads)

        # Chart history (обрезается для отрисовки) + полная история (для отчётов)
        self._upd:  list[int]   = []
        self._avg:  list[float] = []
        self._mx:   list[int]   = []
        self._pg:   list[float] = []
        self._vf:   list[float] = []
        self._ent:  list[float] = []
        self._history_full: list[dict] = []

        self._build_ui()
        self._poll()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Title bar ────────────────────────────────────────────────────────
        bar = tk.Frame(self.container, bg=BG)
        bar.pack(fill=tk.X, padx=10, pady=(6, 2))
        tk.Label(bar, text="Tetris AI  ·  Training Monitor",
                 bg=BG, fg=ACCENT, font=("Consolas", 12, "bold")).pack(side=tk.LEFT)
        self._lbl_status = tk.Label(bar, text="idle", bg=BG, fg=FG2,
                                    font=("Consolas", 9))
        self._lbl_status.pack(side=tk.RIGHT)

        # ── Main columns ─────────────────────────────────────────────────────
        cols = tk.Frame(self.container, bg=BG)
        cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        left = tk.Frame(cols, bg=BG2, width=225)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)
        self._build_left(left)

        center = tk.Frame(cols, bg=BG)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_center(center)

        right = tk.Frame(cols, bg=BG2, width=175)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)
        self._build_right(right)

        # ── Bottom buttons ───────────────────────────────────────────────────
        bot = tk.Frame(self.container, bg=BG3)
        bot.pack(fill=tk.X, padx=10, pady=(2, 8))
        self._build_bottom(bot)

    def _build_left(self, p: tk.Widget) -> None:
        tk.Label(p, text="Hyperparameters", bg=BG2, fg=ACCENT,
                 font=("Consolas", 9, "bold")).pack(pady=(8, 2))
        _sep(p)

        self._hp: dict[str, ParamRow] = {}

        def add(key, label, val, fmt=float):
            self._hp[key] = ParamRow(p, label, val, fmt=fmt)

        # Live-applicable params
        tk.Label(p, text="live (Apply takes effect immediately)",
                 bg=BG2, fg=FG2, font=("Consolas", 7)).pack()
        add("lr",         "LR",          config.PPO_LR)
        add("gamma",      "gamma",       config.PPO_GAMMA)
        add("clip",       "clip eps",    config.PPO_CLIP)
        add("ent_coef",   "ent coef",    config.PPO_ENT_COEF)
        add("vf_coef",    "vf coef",     config.PPO_VF_COEF)
        add("epochs",     "epochs/upd",  config.PPO_EPOCHS,     fmt=int)
        add("batch_size", "batch size",  config.PPO_BATCH_SIZE,  fmt=int)
        add("target",     "target lines",config.PPO_TARGET_SCORE)

        _sep(p)

        # Reward shaping weights (live — пишутся прямо в config, читаются env'ом)
        tk.Label(p, text="reward shaping (live)",
                 bg=BG2, fg=FG2, font=("Consolas", 7)).pack()
        add("w_holes", "w holes",  config.REWARD_W_HOLES)
        add("w_bump",  "w bump",   config.REWARD_W_BUMP)
        add("w_agg",   "w agg_h",  config.REWARD_W_AGG)

        _sep(p)

        # Session params (need restart to apply)
        tk.Label(p, text="session (need restart to apply)",
                 bg=BG2, fg=FG2, font=("Consolas", 7)).pack()
        add("n_envs",    "n envs",      config.PPO_N_ENVS,      fmt=int)
        add("rollout",   "rollout",     config.PPO_ROLLOUT,      fmt=int)
        add("tot_steps", "total steps", config.PPO_TOTAL_STEPS,  fmt=int)

        _sep(p)

        # Autopilot — обучение без присмотра: авто-сейв best, LR-decay на плато,
        # подброс энтропии при коллапсе. Логи действий идут в лог-бокс.
        self._auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            p, text="autopilot (unattended)", variable=self._auto_var,
            bg=BG2, fg=YELLOW, selectcolor=BG3,
            activebackground=BG2, activeforeground=YELLOW,
            font=("Consolas", 8, "bold"), anchor="w",
        ).pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(p, text="auto-save · LR-decay · entropy guard",
                 bg=BG2, fg=FG2, font=("Consolas", 7), anchor="w").pack(
            fill=tk.X, padx=6)

        _sep(p)
        f = tk.Frame(p, bg=BG2)
        f.pack(fill=tk.X, padx=6, pady=6)
        _btn(f, "Apply live", self._apply_overrides, bg=ACCENT, fg=BG).pack(fill=tk.X)

    def _build_center(self, p: tk.Widget) -> None:
        # Matplotlib figure with 2 subplots
        fig = Figure(figsize=(5, 4.2), facecolor=BG)
        self._ax_lines = fig.add_subplot(2, 1, 1, facecolor=BG3)
        self._ax_loss  = fig.add_subplot(2, 1, 2, facecolor=BG3)
        fig.tight_layout(pad=2.0)
        self._fig = fig

        self._canvas = FigureCanvasTkAgg(fig, master=p)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Log box
        lf = tk.Frame(p, bg=BG)
        lf.pack(fill=tk.X, pady=(3, 0))
        self._log = tk.Text(lf, height=5, bg=BG2, fg=FG2,
                            font=("Consolas", 8), state=tk.DISABLED,
                            relief=tk.FLAT, bd=0)
        sb = tk.Scrollbar(lf, command=self._log.yview,
                          bg=BG2, troughcolor=BG3, relief=tk.FLAT)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log.pack(fill=tk.X)

    def _build_right(self, p: tk.Widget) -> None:
        tk.Label(p, text="Status", bg=BG2, fg=ACCENT,
                 font=("Consolas", 9, "bold")).pack(pady=(8, 2))
        _sep(p)

        self._sv: dict[str, tk.StringVar] = {}
        for key, label in [
            ("step",    "step"),
            ("updates", "updates"),
            ("sps",     "SPS"),
            ("avg",     "avg lines"),
            ("max",     "max lines"),
            ("best",    "best avg"),
            ("pg",      "pg loss"),
            ("vf",      "vf loss"),
            ("ent",     "entropy"),
        ]:
            row = tk.Frame(p, bg=BG2)
            row.pack(fill=tk.X, padx=6, pady=2)
            tk.Label(row, text=label, bg=BG2, fg=FG2,
                     font=("Consolas", 8), width=8, anchor="w").pack(side=tk.LEFT)
            v = tk.StringVar(value="—")
            tk.Label(row, textvariable=v, bg=BG2, fg=FG,
                     font=("Consolas", 8)).pack(side=tk.RIGHT)
            self._sv[key] = v

        _sep(p)
        tk.Label(p, text=f"device: {get_device()}", bg=BG2, fg=FG2,
                 font=("Consolas", 7)).pack(pady=4)

    def _build_bottom(self, p: tk.Widget) -> None:
        self._btn_start = _btn(p, "Start", self._start, bg=GREEN, fg=BG)
        self._btn_start.pack(side=tk.LEFT, padx=(8, 4), pady=7)

        self._btn_stop = _btn(p, "Stop", self._stop, bg=RED, fg=BG,
                              state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT, padx=4, pady=7)

        self._btn_save = _btn(p, "Save Model", self._save, bg=PURPLE, fg=BG,
                              state=tk.DISABLED)
        self._btn_save.pack(side=tk.LEFT, padx=4, pady=7)

        _btn(p, "Save Charts", self._save_charts, bg=ACCENT, fg=BG).pack(
            side=tk.LEFT, padx=4, pady=7)

        _btn(p, "Clear Charts", self._clear_charts, bg=BG, fg=FG2).pack(
            side=tk.LEFT, padx=4, pady=7)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        hp = self._read_hp(full=True)
        if hp is None:
            return
        if not self._apply_shaping():   # зафиксировать веса shaping в config
            return

        autopilot = self._auto_var.get()

        self._stop_evt.clear()
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._btn_save.config(state=tk.DISABLED)
        self._lbl_status.config(text="training", fg=GREEN)
        self._run_dir = runs_mod.create_run_dir('train')
        self._log_write("Training started." +
                        ("  [autopilot ON]" if autopilot else ""))
        self._log_write(f"Run dir: {self._run_dir}")

        existing_model = self._model   # resume from previous run if present

        def _run():
            from ppo_trainer import train_ppo
            result = train_ppo(
                total_steps=hp["tot_steps"],
                n_envs=hp["n_envs"],
                rollout=hp["rollout"],
                epochs=hp["epochs"],
                batch_size=hp["batch_size"],
                lr=hp["lr"],
                gamma=hp["gamma"],
                clip=hp["clip"],
                vf_coef=hp["vf_coef"],
                ent_coef=hp["ent_coef"],
                target_score=hp["target"],
                model=existing_model,
                should_stop=self._stop_evt.is_set,
                on_update=lambda rec: self._q.put(("upd", rec)),
                on_log=lambda msg: self._q.put(("log", msg)),
                overrides=self._overrides,
                autopilot=autopilot,
                verbose=False,
            )
            self._q.put(("done", result))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        self._stop_evt.set()
        self._btn_stop.config(state=tk.DISABLED)
        self._log_write("Stop requested…")

    def _save(self) -> None:
        if self._model is None:
            return
        path = storage.save_oracle(self._model)
        self._log_write(f"Saved to {path}")
        if getattr(self, '_run_dir', None):
            runs_mod.copy_into(self._run_dir, path)
            self._log_write(f"Archived -> {self._run_dir}")
        messagebox.showinfo("Saved", f"Model saved:\n{path}")

    def _save_charts(self) -> None:
        """История обучения → results/ (PNG learning curve + JSON истории)."""
        if not self._history_full:
            messagebox.showinfo("Save Charts", "История пуста — нечего сохранять.")
            return
        png = reports.plot_ppo_history(self._history_full)
        js = reports.save_ppo_history(self._history_full)
        self._log_write(f"Charts saved -> {png}")
        self._log_write(f"History saved -> {js}")

    def _apply_overrides(self) -> None:
        hp = self._read_hp(full=False)
        if hp is None:
            return
        self._overrides.update(hp)
        self._apply_shaping()
        self._log_write(
            f"Overrides applied — lr={hp['lr']:.2e}  clip={hp['clip']}  "
            f"ent={hp['ent_coef']}  vf={hp['vf_coef']}  "
            f"ep={hp['epochs']}  bs={hp['batch_size']}"
        )

    def _apply_shaping(self) -> bool:
        """Записать веса reward-shaping прямо в config (env читает их вживую)."""
        names = {"w_holes": "REWARD_W_HOLES", "w_bump": "REWARD_W_BUMP",
                 "w_agg": "REWARD_W_AGG"}
        applied = {}
        for key, cfg_name in names.items():
            v = self._hp[key].get()
            if v is None:
                messagebox.showerror("Bad value", f"Invalid value for: {key}")
                return False
            setattr(config, cfg_name, v)
            applied[key] = v
        self._log_write(
            f"Shaping applied — holes={applied['w_holes']} "
            f"bump={applied['w_bump']} agg={applied['w_agg']}"
        )
        return True

    def _clear_charts(self) -> None:
        self._upd.clear(); self._avg.clear(); self._mx.clear()
        self._pg.clear();  self._vf.clear(); self._ent.clear()
        self._redraw()

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll(self) -> None:
        dirty = False
        try:
            while True:
                kind, data = self._q.get_nowait()
                if kind == "upd":
                    self._absorb(data)
                    dirty = True
                elif kind == "log":
                    self._log_write(f"[auto] {data}")
                elif kind == "done":
                    self._on_done(data)
        except queue.Empty:
            pass
        if dirty:
            self._redraw()
        self.root.after(self._POLL_MS, self._poll)

    def _absorb(self, rec: dict) -> None:
        """Принять один update-record из очереди."""
        self._history_full.append(rec)
        self._upd.append(rec["update"])
        self._avg.append(rec["avg_lines"])
        self._mx.append(rec["max_lines"])
        self._pg.append(rec["pg_loss"])
        self._vf.append(rec["vf_loss"])
        self._ent.append(rec["entropy"])

        if len(self._upd) > self._CHART_MAX:
            self._upd = self._upd[-self._CHART_MAX:]
            self._avg = self._avg[-self._CHART_MAX:]
            self._mx  = self._mx[-self._CHART_MAX:]
            self._pg  = self._pg[-self._CHART_MAX:]
            self._vf  = self._vf[-self._CHART_MAX:]
            self._ent = self._ent[-self._CHART_MAX:]

        best = max(self._avg) if self._avg else 0.0
        sv = self._sv
        sv["step"].set(f"{rec['global_step']:,}")
        sv["updates"].set(str(rec["update"]))
        sv["sps"].set(f"{rec['sps']:,.0f}")
        sv["avg"].set(f"{rec['avg_lines']:.2f}")
        sv["max"].set(str(rec["max_lines"]))
        sv["best"].set(f"{best:.2f}")
        sv["pg"].set(f"{rec['pg_loss']:+.4f}")
        sv["vf"].set(f"{rec['vf_loss']:.4f}")
        sv["ent"].set(f"{rec['entropy']:.4f}")

    def _on_done(self, result: dict) -> None:
        self._model = result["model"]
        self._btn_start.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)
        self._btn_save.config(state=tk.NORMAL)
        reason = result.get("stop_reason") or "budget exhausted"
        self._lbl_status.config(text=f"done ({reason})", fg=FG2)
        self._log_write(
            f"Done — steps={result['total_steps']:,}  "
            f"best={result['best_avg_lines']:.2f} lines  "
            f"time={result['elapsed'] / 60:.1f} min"
        )
        # Полный слепок в run-папку: модель + история + графики + summary.
        # Канонический best_ppo.pt НЕ трогаем — он по кнопке Save Model.
        run_dir = getattr(self, '_run_dir', None)
        if run_dir:
            try:
                meta = {'total_steps': result['total_steps'],
                        'best_avg_lines': result['best_avg_lines'],
                        'stop_reason': reason,
                        'elapsed_min': round(result['elapsed'] / 60, 1)}
                storage.save_oracle(
                    result['model'],
                    path=os.path.join(run_dir, 'model.pt'), meta=meta)
                if result.get('history'):
                    hist = reports.save_ppo_history(result['history'], meta=meta)
                    png = reports.plot_ppo_history(result['history'])
                    runs_mod.copy_into(run_dir, hist, png)
                runs_mod.save_summary(run_dir, meta)
                self._log_write(f"Run archived -> {run_dir}")
            except Exception as exc:  # noqa: BLE001
                self._log_write(f"run archive failed: {exc!r}")

    # ── Charts ────────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        xs = self._upd
        ax1, ax2 = self._ax_lines, self._ax_loss

        ax1.cla()
        ax1.set_facecolor(BG3)
        ax1.set_title("Lines cleared  (avg / max)", color=FG2, fontsize=8, pad=2)
        if xs:
            ax1.plot(xs, self._avg, color=GREEN,  lw=1.4, label="avg")
            ax1.plot(xs, self._mx,  color=ACCENT, lw=0.9, alpha=0.55, label="max")
            ax1.legend(fontsize=7, facecolor=BG2, edgecolor=BG3,
                       labelcolor=FG2, loc="upper left")
        _style(ax1)

        ax2.cla()
        ax2.set_facecolor(BG3)
        ax2.set_title("Losses  (pg / vf / entropy)", color=FG2, fontsize=8, pad=2)
        if xs:
            ax2.plot(xs, self._pg,  color=RED,    lw=1.0, label="pg")
            ax2.plot(xs, self._vf,  color=YELLOW, lw=1.0, alpha=0.85, label="vf")
            ax2.plot(xs, self._ent, color=PURPLE, lw=1.0, alpha=0.85, label="entropy")
            ax2.legend(fontsize=7, facecolor=BG2, edgecolor=BG3,
                       labelcolor=FG2, loc="upper right")
        _style(ax2)

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw_idle()

    # ── Log ──────────────────────────────────────────────────────────────────

    def _log_write(self, msg: str) -> None:
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, f"» {msg}\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_hp(self, full: bool) -> dict | None:
        keys = ["lr", "gamma", "clip", "ent_coef", "vf_coef",
                "epochs", "batch_size", "target"]
        if full:
            keys += ["n_envs", "rollout", "tot_steps"]
        out = {}
        for k in keys:
            v = self._hp[k].get()
            if v is None:
                messagebox.showerror("Bad value", f"Invalid value for: {k}")
                return None
            out[k] = v
        return out


# ── Distillation tab ──────────────────────────────────────────────────────────

# v2: одна формула (placement-value), а не 6 формул по действиям.
ACTION_NAMES = ["PLACE"]
ACTION_COLORS = [GREEN]
N_FORMULAS = 1


class DistillTab:
    """Вкладка EML-дистилляции: оракул → формула (DATA + GAME фазы)."""

    _POLL_MS = 250

    def __init__(self, root: tk.Tk, container: tk.Widget):
        self.root = root
        self.container = container

        self._q: queue.Queue = queue.Queue()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

        # История графиков.
        self._data_fit: dict[int, tuple[list, list]] = {
            a: ([], []) for a in range(N_FORMULAS)}
        self._joint_x: list[int] = []
        self._joint_y: list[float] = []

        self._build_ui()
        self._poll()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        bar = tk.Frame(self.container, bg=BG)
        bar.pack(fill=tk.X, padx=10, pady=(6, 2))
        tk.Label(bar, text="Tetris AI  ·  EML Distillation",
                 bg=BG, fg=PURPLE, font=("Consolas", 12, "bold")).pack(side=tk.LEFT)
        self._lbl_status = tk.Label(bar, text="idle", bg=BG, fg=FG2,
                                    font=("Consolas", 9))
        self._lbl_status.pack(side=tk.RIGHT)

        cols = tk.Frame(self.container, bg=BG)
        cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        left = tk.Frame(cols, bg=BG2, width=225)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)
        self._build_left(left)

        center = tk.Frame(cols, bg=BG)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_center(center)

        right = tk.Frame(cols, bg=BG2, width=185)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)
        self._build_right(right)

        bot = tk.Frame(self.container, bg=BG3)
        bot.pack(fill=tk.X, padx=10, pady=(2, 8))
        self._build_bottom(bot)

    def _build_left(self, p: tk.Widget) -> None:
        tk.Label(p, text="Distill params", bg=BG2, fg=PURPLE,
                 font=("Consolas", 9, "bold")).pack(pady=(8, 2))
        _sep(p)

        self._hp: dict[str, ParamRow] = {}

        def add(key, label, val, fmt=int):
            self._hp[key] = ParamRow(p, label, val, fmt=fmt)

        tk.Label(p, text="dataset", bg=BG2, fg=FG2,
                 font=("Consolas", 7)).pack()
        add("episodes", "episodes", config.EML_DATASET_EPISODES)

        _sep(p)
        tk.Label(p, text="batch (variants>1 = пачка формул)", bg=BG2, fg=FG2,
                 font=("Consolas", 7)).pack()
        add("variants", "variants", config.EML_BATCH_VARIANTS)
        self._hp["spread"] = ParamRow(p, "spread ±", config.EML_BATCH_SPREAD,
                                      fmt=float)

        _sep(p)
        tk.Label(p, text="DATA phase (regression)", bg=BG2, fg=FG2,
                 font=("Consolas", 7)).pack()
        add("data_gen", "generations", config.EML_GENERATIONS)
        add("data_pop", "population", config.EML_POPULATION)

        _sep(p)
        tk.Label(p, text="GAME phase (joint)", bg=BG2, fg=FG2,
                 font=("Consolas", 7)).pack()
        add("joint_gen", "generations", config.EML_JOINT_GENERATIONS)
        add("joint_pop", "population", config.EML_JOINT_POPULATION)
        add("joint_games", "games/eval", config.EML_INGAME_GAMES)

        _sep(p)
        # Depth penalty (OptionMenu).
        dp_row = tk.Frame(p, bg=BG2)
        dp_row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(dp_row, text="depth pen.", bg=BG2, fg=FG2,
                 font=("Consolas", 8), width=13, anchor="w").pack(side=tk.LEFT)
        self._dp_var = tk.StringVar(value="medium")
        om = tk.OptionMenu(dp_row, self._dp_var, "weak", "medium", "strong")
        om.config(bg=BG3, fg=FG, font=("Consolas", 8), relief=tk.FLAT,
                  highlightthickness=0, activebackground=BG3, width=8)
        om["menu"].config(bg=BG3, fg=FG, font=("Consolas", 8))
        om.pack(side=tk.LEFT, padx=(2, 0))

        # Checkboxes.
        self._reuse_var = tk.BooleanVar(value=False)
        self._joint_var = tk.BooleanVar(value=True)
        for var, text in [(self._reuse_var, "reuse cached dataset"),
                          (self._joint_var, "run GAME phase")]:
            tk.Checkbutton(
                p, text=text, variable=var, bg=BG2, fg=FG2,
                selectcolor=BG3, activebackground=BG2, activeforeground=FG,
                font=("Consolas", 8), anchor="w",
            ).pack(fill=tk.X, padx=6, pady=1)

        _sep(p)
        tk.Label(p, text="formulas auto-saved to\nmodels/best_eml.json",
                 bg=BG2, fg=FG2, font=("Consolas", 7), justify="left").pack(
            padx=6, pady=4, anchor="w")

    def _build_center(self, p: tk.Widget) -> None:
        fig = Figure(figsize=(5, 4.2), facecolor=BG)
        self._ax_data = fig.add_subplot(2, 1, 1, facecolor=BG3)
        self._ax_joint = fig.add_subplot(2, 1, 2, facecolor=BG3)
        fig.tight_layout(pad=2.0)
        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=p)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        lf = tk.Frame(p, bg=BG)
        lf.pack(fill=tk.X, pady=(3, 0))
        self._log = tk.Text(lf, height=6, bg=BG2, fg=FG2,
                            font=("Consolas", 8), state=tk.DISABLED,
                            relief=tk.FLAT, bd=0)
        sb = tk.Scrollbar(lf, command=self._log.yview,
                          bg=BG2, troughcolor=BG3, relief=tk.FLAT)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log.pack(fill=tk.X)

    def _build_right(self, p: tk.Widget) -> None:
        tk.Label(p, text="Status", bg=BG2, fg=PURPLE,
                 font=("Consolas", 9, "bold")).pack(pady=(8, 2))
        _sep(p)
        self._sv: dict[str, tk.StringVar] = {}
        for key, label in [
            ("phase",   "phase"),
            ("variant", "variant"),
            ("samples", "samples"),
            ("dgen",    "DATA gen"),
            ("dfit",    "DATA fit"),
            ("jgen",    "GAME gen"),
            ("jlines",  "GAME lines"),
            ("eml",     "EML lines"),
            ("oracle",  "oracle lines"),
            ("ratio",   "EML/oracle"),
            ("size",    "AST size"),
        ]:
            row = tk.Frame(p, bg=BG2)
            row.pack(fill=tk.X, padx=6, pady=2)
            tk.Label(row, text=label, bg=BG2, fg=FG2,
                     font=("Consolas", 8), width=11, anchor="w").pack(side=tk.LEFT)
            v = tk.StringVar(value="—")
            tk.Label(row, textvariable=v, bg=BG2, fg=FG,
                     font=("Consolas", 8)).pack(side=tk.RIGHT)
            self._sv[key] = v

        _sep(p)
        tk.Label(p, text=f"device: {get_device()}", bg=BG2, fg=FG2,
                 font=("Consolas", 7)).pack(pady=4)

    def _build_bottom(self, p: tk.Widget) -> None:
        self._btn_start = _btn(p, "Distill", self._start, bg=PURPLE, fg=BG)
        self._btn_start.pack(side=tk.LEFT, padx=(8, 4), pady=7)
        self._btn_stop = _btn(p, "Stop", self._stop, bg=RED, fg=BG,
                              state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT, padx=4, pady=7)
        _btn(p, "Clear Charts", self._clear_charts, bg=BG, fg=FG2).pack(
            side=tk.LEFT, padx=4, pady=7)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        hp = self._read_hp()
        if hp is None:
            return

        self._stop_evt.clear()
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._lbl_status.config(text="running", fg=GREEN)
        self._clear_charts()
        self._log_write("Distillation started.")

        dp_name = self._dp_var.get()
        reuse = self._reuse_var.get()
        joint = self._joint_var.get()
        n_variants = max(1, hp.get("variants", 1))
        spread = max(0.0, hp.get("spread", 0.0))

        def _run():
            from pipeline import full_distill, batch_distill
            # GAME-фаза управляется через config (читается при None-параметрах).
            config.EML_JOINT_GENERATIONS = hp["joint_gen"]
            config.EML_JOINT_POPULATION = hp["joint_pop"]
            config.EML_INGAME_GAMES = hp["joint_games"]
            try:
                common = dict(
                    n_episodes=hp["episodes"],
                    reuse_data=reuse,
                    data_generations=hp["data_gen"],
                    data_population=hp["data_pop"],
                    joint=joint,
                    eval_games=hp["joint_games"],
                    on_event=lambda ev: self._q.put(ev),
                    should_stop=self._stop_evt.is_set,
                    verbose=False,
                )
                if n_variants > 1:
                    batch_distill(n_variants=n_variants, spread=spread,
                                  **common)
                else:
                    full_distill(depth_penalty_name=dp_name, **common)
            except Exception as exc:  # noqa: BLE001 — показать ошибку в GUI
                self._q.put({'type': 'error', 'msg': repr(exc)})

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        self._stop_evt.set()
        self._btn_stop.config(state=tk.DISABLED)
        self._log_write("Stop requested…")

    def _clear_charts(self) -> None:
        self._data_fit = {a: ([], []) for a in range(N_FORMULAS)}
        self._joint_x.clear()
        self._joint_y.clear()
        self._redraw()

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll(self) -> None:
        dirty = False
        try:
            while True:
                ev = self._q.get_nowait()
                dirty |= self._absorb(ev)
        except queue.Empty:
            pass
        if dirty:
            self._redraw()
        self.root.after(self._POLL_MS, self._poll)

    def _absorb(self, ev: dict) -> bool:
        """Обработать одно событие. Возвращает True, если график нужно перерисовать."""
        t = ev.get('type')
        sv = self._sv

        if t == 'log':
            self._log_write(ev['msg'])
        elif t == 'phase':
            sv['phase'].set(ev['phase'])
            self._log_write(f"— phase: {ev['phase']}")
        elif t == 'collect':
            sv['phase'].set("dataset")
            sv['samples'].set(f"{ev['samples']:,} ({ev['episode']}/{ev['total']})")
        elif t == 'dataset_ready':
            sv['samples'].set(f"{ev['samples']:,}")
            self._log_write(f"dataset ready: {ev['samples']:,} samples, "
                            f"counts={ev['counts']}")
        elif t == 'variant_start':
            sv['variant'].set(f"{ev['variant'] + 1}/{ev['total']}")
            p = ev['params']
            self._log_write(
                f"— variant {ev['variant'] + 1}/{ev['total']}: "
                f"dp={p['depth_penalty']} gen={p['generations']} "
                f"pop={p['population']} depth<={p['max_depth']} "
                f"seed={p['seed']}")
            # Каждый вариант — свои кривые.
            self._data_fit = {a: ([], []) for a in range(N_FORMULAS)}
            self._joint_x.clear()
            self._joint_y.clear()
            return True
        elif t == 'variant_done':
            self._log_write(
                f"  variant {ev['variant'] + 1}/{ev['total']} -> "
                f"{ev['final_lines']:.1f} lines, AST {ev['size']}: "
                f"{ev['formula']}")
        elif t == 'data_gen':
            xs, ys = self._data_fit[ev['action']]
            xs.append(ev['gen'])
            ys.append(ev['fitness'])
            sv['dgen'].set(f"{ev['gen'] + 1}/{ev['generations']}")
            sv['dfit'].set(f"{ev['fitness']:.2f}")
            return True
        elif t == 'data_action_done':
            self._log_write(
                f"  f[{ev['name']}] D{ev['depth']} S{ev['size']} "
                f"V{ev['nvars']}  fit={ev['fitness']:.2f}")
        elif t == 'base_score':
            self._log_write(f"DATA joint score: {ev['base_lines']:.2f} lines/game")
            sv['eml'].set(f"{ev['base_lines']:.2f}")
        elif t == 'joint_gen':
            self._joint_x.append(ev['gen'])
            self._joint_y.append(ev['lines'])
            sv['jgen'].set(f"{ev['gen'] + 1}/{ev['generations']}")
            sv['jlines'].set(f"{ev['lines']:.2f}")
            return True
        elif t == 'done':
            self._on_done(ev)
        elif t == 'cancelled':
            self._on_finish("cancelled", RED)
            self._log_write("Cancelled.")
        elif t == 'error':
            self._on_finish("error", RED)
            self._log_write(f"ERROR: {ev['msg']}")
            messagebox.showerror("Distillation error", ev['msg'])
        return False

    def _on_done(self, ev: dict) -> None:
        sv = self._sv
        sv['eml'].set(f"{ev['eml_lines']:.2f}")
        sv['oracle'].set(f"{ev['oracle_lines']:.2f}")
        sv['ratio'].set(f"{ev['ratio_pct']:.1f}%")
        sv['size'].set(f"{ev['total_size']}")
        self._log_write(
            f"DONE — EML {ev['eml_lines']:.2f} vs oracle {ev['oracle_lines']:.2f} "
            f"({ev['ratio_pct']:.1f}%), AST size {ev['total_size']}")
        if ev.get('run_dir'):
            self._log_write(f"Run archive -> {ev['run_dir']}")
        for a, f in enumerate(ev['formulas']):
            name = ACTION_NAMES[a] if a < len(ACTION_NAMES) else f"f{a}"
            self._log_write(f"  f[{name}] = {f[:60]}")
        self._log_write(f"Saved -> {ev['path']}")
        self._on_finish("done", FG2)

    def _on_finish(self, status: str, color: str) -> None:
        self._lbl_status.config(text=status, fg=color)
        self._btn_start.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)

    # ── Charts ────────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        ax1, ax2 = self._ax_data, self._ax_joint

        ax1.cla()
        ax1.set_facecolor(BG3)
        ax1.set_title("DATA fitness", color=FG2, fontsize=8, pad=2)
        has_data = False
        for a in range(N_FORMULAS):
            xs, ys = self._data_fit[a]
            if xs:
                has_data = True
                ax1.plot(xs, ys, color=ACTION_COLORS[a], lw=1.1,
                         label=ACTION_NAMES[a])
        if has_data:
            ax1.legend(fontsize=6, facecolor=BG2, edgecolor=BG3,
                       labelcolor=FG2, loc="upper left", ncol=3)
        _style(ax1)

        ax2.cla()
        ax2.set_facecolor(BG3)
        ax2.set_title("JOINT score  (lines/game)", color=FG2, fontsize=8, pad=2)
        if self._joint_x:
            ax2.plot(self._joint_x, self._joint_y, color=GREEN, lw=1.4)
        _style(ax2)

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw_idle()

    # ── Log / helpers ───────────────────────────────────────────────────────

    def _log_write(self, msg: str) -> None:
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, f"» {msg}\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _read_hp(self) -> dict | None:
        out = {}
        for k in ["episodes", "data_gen", "data_pop",
                  "joint_gen", "joint_pop", "joint_games", "variants"]:
            v = self._hp[k].get()
            if v is None or v <= 0:
                messagebox.showerror("Bad value", f"Invalid value for: {k}")
                return None
            out[k] = v
        spread = self._hp["spread"].get()
        if spread is None or spread < 0:
            messagebox.showerror("Bad value", "Invalid value for: spread")
            return None
        out["spread"] = spread
        return out


# ── Play tab ──────────────────────────────────────────────────────────────────

_CELL = 24          # пикселей на клетку доски
_PIECE_COLORS = ["#89dceb", "#f9e2af", "#cba6f7", "#a6e3a1",
                 "#f38ba8", "#89b4fa", "#fab387"]   # I O T S Z J L


class PlayTab:
    """Вкладка Play: смотреть, как играет обученный оракул или EML-формула."""

    def __init__(self, root: tk.Tk, container: tk.Widget):
        self.root = root
        self.container = container

        # Play — без капа постановок (max_placements=0): игра до game over.
        self._env = TetrisEnv(seed=0, max_placements=0)
        self._obs = self._env.reset()
        self._policy = None            # callable(obs) -> action | -1
        self._policy_name = "—"
        self._playing = False
        self._anim: dict | None = None   # {'a','rot','x','y','ty'}
        self._game_no = 0
        self._pieces = 0
        self._tick_scheduled = False
        self._duel: dict | None = None   # состояние duel-режима (2 доски)
        self._lat_last = 0.0             # µs последнего решения
        self._lat_sum = 0.0              # накопление для среднего за игру
        self._lat_n = 0

        self._build_ui()
        self._redraw_board()

    # ── Build ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        bar = tk.Frame(self.container, bg=BG)
        bar.pack(fill=tk.X, padx=10, pady=(6, 2))
        tk.Label(bar, text="Tetris AI  ·  Watch it play",
                 bg=BG, fg=GREEN, font=("Consolas", 12, "bold")).pack(side=tk.LEFT)
        self._lbl_status = tk.Label(bar, text="idle", bg=BG, fg=FG2,
                                    font=("Consolas", 9))
        self._lbl_status.pack(side=tk.RIGHT)

        cols = tk.Frame(self.container, bg=BG)
        cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # ── Left: выбор модели и управление ──────────────────────────────
        left = tk.Frame(cols, bg=BG2, width=250)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)

        tk.Label(left, text="Model", bg=BG2, fg=GREEN,
                 font=("Consolas", 9, "bold")).pack(pady=(8, 2))
        _sep(left)

        self._src_var = tk.StringVar(value="oracle")
        for val, text in [("oracle", "CNN oracle (best_ppo.pt)"),
                          ("eml", "EML formula (best_eml.json)"),
                          ("file", "custom file…")]:
            tk.Radiobutton(
                left, text=text, variable=self._src_var, value=val,
                bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
                activeforeground=FG, font=("Consolas", 8), anchor="w",
            ).pack(fill=tk.X, padx=6)

        self._duel_var = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="duel: oracle vs EML (same seed)",
                       variable=self._duel_var, bg=BG2, fg=PURPLE,
                       selectcolor=BG3, activebackground=BG2,
                       activeforeground=PURPLE,
                       font=("Consolas", 8, "bold"), anchor="w").pack(
            fill=tk.X, padx=6, pady=(2, 0))

        f = tk.Frame(left, bg=BG2)
        f.pack(fill=tk.X, padx=6, pady=(4, 2))
        _btn(f, "Load model", self._load_model, bg=ACCENT, fg=BG).pack(fill=tk.X)

        _sep(left)
        self._pr_seed = ParamRow(left, "seed", 0, fmt=int)
        self._pr_speed = ParamRow(left, "anim ms/row", 30, fmt=int)

        self._instant_var = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="instant (no fall anim)",
                       variable=self._instant_var, bg=BG2, fg=FG2,
                       selectcolor=BG3, activebackground=BG2,
                       font=("Consolas", 8), anchor="w").pack(fill=tk.X, padx=6)
        self._restart_var = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="auto-restart after game over",
                       variable=self._restart_var, bg=BG2, fg=FG2,
                       selectcolor=BG3, activebackground=BG2,
                       font=("Consolas", 8), anchor="w").pack(fill=tk.X, padx=6)

        _sep(left)
        bf = tk.Frame(left, bg=BG2)
        bf.pack(fill=tk.X, padx=6, pady=4)
        self._btn_play = _btn(bf, "▶ Play", self._toggle_play, bg=GREEN, fg=BG)
        self._btn_play.pack(fill=tk.X, pady=2)
        _btn(bf, "Step (1 piece)", self._step_once, bg=BG3, fg=FG).pack(
            fill=tk.X, pady=2)
        _btn(bf, "Reset game", self._reset_game, bg=RED, fg=BG).pack(
            fill=tk.X, pady=2)

        _sep(left)
        tk.Label(left, text=f"device: {get_device()}", bg=BG2, fg=FG2,
                 font=("Consolas", 7)).pack(pady=4)

        # ── Center: доска ─────────────────────────────────────────────────
        center = tk.Frame(cols, bg=BG)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._canvas = tk.Canvas(
            center, width=config.BOARD_W * _CELL,
            height=config.BOARD_H_TOTAL * _CELL,
            bg=BG3, highlightthickness=1, highlightbackground=BG2)
        self._canvas.pack(pady=6)

        # ── Right: статус ─────────────────────────────────────────────────
        right = tk.Frame(cols, bg=BG2, width=200)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)
        tk.Label(right, text="Status", bg=BG2, fg=GREEN,
                 font=("Consolas", 9, "bold")).pack(pady=(8, 2))
        _sep(right)
        self._sv: dict[str, tk.StringVar] = {}
        for key, label in [("model", "model"), ("game", "game #"),
                           ("lines", "lines"), ("pieces", "pieces"),
                           ("next", "next piece"),
                           ("lat", "µs/move"), ("latavg", "µs avg")]:
            row = tk.Frame(right, bg=BG2)
            row.pack(fill=tk.X, padx=6, pady=2)
            tk.Label(row, text=label, bg=BG2, fg=FG2,
                     font=("Consolas", 8), width=10, anchor="w").pack(side=tk.LEFT)
            v = tk.StringVar(value="—")
            tk.Label(row, textvariable=v, bg=BG2, fg=FG,
                     font=("Consolas", 8)).pack(side=tk.RIGHT)
            self._sv[key] = v
        _sep(right)
        tk.Label(right, text="finished games", bg=BG2, fg=FG2,
                 font=("Consolas", 8)).pack()
        self._games_log = tk.Text(right, height=14, width=22, bg=BG3, fg=FG2,
                                  font=("Consolas", 8), state=tk.DISABLED,
                                  relief=tk.FLAT, bd=0)
        self._games_log.pack(padx=6, pady=4, fill=tk.BOTH, expand=True)

    # ── Model loading ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._duel_var.get():
            self._load_duel()
            return
        src = self._src_var.get()
        try:
            if src == "oracle":
                self._load_oracle(os.path.join(config.MODELS_DIR, 'best_ppo.pt'))
            elif src == "eml":
                self._load_eml(os.path.join(config.MODELS_DIR, 'best_eml.json'))
            else:
                path = filedialog.askopenfilename(
                    initialdir=config.MODELS_DIR,
                    filetypes=[("Model", "*.pt *.json")])
                if not path:
                    return
                if path.endswith('.pt'):
                    self._load_oracle(path)
                else:
                    self._load_eml(path)
        except Exception as exc:  # noqa: BLE001 — показать пользователю
            messagebox.showerror("Load failed", repr(exc))
            return
        self._duel = None
        self._canvas.config(width=config.BOARD_W * _CELL)
        self._sv['model'].set(self._policy_name)
        self._lbl_status.config(text=f"loaded: {self._policy_name}", fg=GREEN)
        self._reset_game()

    # ── Duel: оракул vs формула на двух досках с одним сидом ─────────────

    _DUEL_GAP = 3 * _CELL     # зазор между досками

    def _load_duel(self) -> None:
        try:
            self._load_oracle(os.path.join(config.MODELS_DIR, 'best_ppo.pt'))
            pol_a, name_a = self._policy, self._policy_name
            self._load_eml(os.path.join(config.MODELS_DIR, 'best_eml.json'))
            pol_b, name_b = self._policy, self._policy_name
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load failed", repr(exc))
            return
        self._policy = None
        self._duel = {'pols': [pol_a, pol_b], 'names': [name_a, name_b],
                      'envs': [], 'obs': [], 'pieces': [0, 0],
                      'done': [False, False]}
        self._canvas.config(
            width=config.BOARD_W * _CELL * 2 + self._DUEL_GAP)
        self._sv['model'].set("CNN vs EML")
        self._lbl_status.config(text=f"duel: {name_a} vs {name_b}", fg=PURPLE)
        self._duel_reset()

    def _duel_reset(self) -> None:
        seed = self._pr_seed.get() or 0
        self._game_no += 1
        d = self._duel
        d['envs'] = [TetrisEnv(seed=seed + self._game_no - 1,
                               max_placements=0) for _ in range(2)]
        d['obs'] = [e.reset() for e in d['envs']]
        d['pieces'] = [0, 0]
        d['done'] = [False, False]
        d['lat'] = [[0.0, 0], [0.0, 0]]     # [сумма µs, ходов] на сторону
        self._sv['game'].set(str(self._game_no))
        self._redraw_board()

    def _duel_tick(self) -> None:
        d = self._duel
        last_us = [0.0, 0.0]
        for i in range(2):
            if d['done'][i]:
                continue
            t0 = time.perf_counter()
            a = d['pols'][i](d['obs'][i])
            last_us[i] = (time.perf_counter() - t0) * 1e6
            d['lat'][i][0] += last_us[i]
            d['lat'][i][1] += 1
            if a < 0:
                d['done'][i] = True
                continue
            d['obs'][i], _r, done, _info = d['envs'][i].step_placement(a)
            d['pieces'][i] += 1
            d['done'][i] = done
        self._sv['lines'].set(
            f"{d['envs'][0].score} vs {d['envs'][1].score}")
        self._sv['pieces'].set(f"{d['pieces'][0]} vs {d['pieces'][1]}")
        self._sv['lat'].set(f"{last_us[0]:,.0f} vs {last_us[1]:,.0f}")
        avgs = [(s / n if n else 0.0) for (s, n) in d['lat']]
        self._sv['latavg'].set(f"{avgs[0]:,.0f} vs {avgs[1]:,.0f}")
        if all(d['done']):
            s0, s1 = d['envs'][0].score, d['envs'][1].score
            winner = "CNN" if s0 > s1 else ("EML" if s1 > s0 else "draw")
            self._games_write(f"#{self._game_no}: CNN {s0} vs EML {s1}"
                              f"  [{winner}]")
            if self._restart_var.get() and self._playing:
                self._duel_reset()
            else:
                self._playing = False
                self._btn_play.config(text="▶ Play")
                self._lbl_status.config(text="duel over", fg=RED)

    def _load_oracle(self, path: str) -> None:
        import torch
        device = get_device()
        model = TetrisCNN().to(device)
        storage.load_oracle(model, path, device=device)
        model.eval()

        def _policy(obs):
            grid, scalars, afeats, mask = obs_to_tensors(obs, device)
            with torch.no_grad():
                logits = model.get_logits(grid, scalars, afeats, mask)
            return int(torch.argmax(logits, dim=-1).item())

        self._policy = _policy
        self._policy_name = os.path.basename(path)

    def _load_eml(self, path: str) -> None:
        trees, _meta = storage.load_formulas(EMLNode, path)
        _check_formula_compat(trees)
        pol = EMLPolicy(trees)

        def _policy(obs):
            return pol.choose(obs['mask'], obs['afeats'], obs['scalars'])

        self._policy = _policy
        self._policy_name = os.path.basename(path)

    # ── Game control ─────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self._policy is None and self._duel is None:
            messagebox.showinfo("Play", "Сначала загрузите модель (Load model).")
            return
        self._playing = not self._playing
        self._btn_play.config(text="⏸ Pause" if self._playing else "▶ Play")
        self._lbl_status.config(
            text="playing" if self._playing else "paused",
            fg=GREEN if self._playing else FG2)
        if self._playing:
            self._schedule_tick()

    def _step_once(self) -> None:
        if self._policy is None and self._duel is None:
            messagebox.showinfo("Play", "Сначала загрузите модель (Load model).")
            return
        self._playing = False
        self._btn_play.config(text="▶ Play")
        if self._duel is not None:
            self._duel_tick()
        else:
            self._place_piece_instant()
        self._redraw_board()

    def _reset_game(self) -> None:
        if self._duel is not None:
            self._duel_reset()
            return
        seed = self._pr_seed.get()
        self._env = TetrisEnv(seed=seed if seed is not None else 0,
                              max_placements=0)
        self._obs = self._env.reset()
        self._anim = None
        self._pieces = 0
        self._reset_latency()
        self._game_no += 1
        self._sv['game'].set(str(self._game_no))
        self._sv['lines'].set("0")
        self._sv['pieces'].set("0")
        self._redraw_board()

    # ── Latency решения (время/ресурсы на ход) ───────────────────────────

    def _decide(self, policy, obs) -> int:
        """Вызов политики с замером времени решения (µs)."""
        t0 = time.perf_counter()
        a = policy(obs)
        dt_us = (time.perf_counter() - t0) * 1e6
        self._lat_last = dt_us
        self._lat_sum += dt_us
        self._lat_n += 1
        self._sv['lat'].set(f"{dt_us:,.0f}")
        self._sv['latavg'].set(f"{self._lat_sum / self._lat_n:,.0f}")
        return a

    def _reset_latency(self) -> None:
        self._lat_sum = 0.0
        self._lat_n = 0

    # ── Tick / animation ─────────────────────────────────────────────────

    def _schedule_tick(self) -> None:
        if self._tick_scheduled:
            return
        self._tick_scheduled = True
        ms = self._pr_speed.get() or 30
        self.root.after(max(5, int(ms)), self._tick)

    def _tick(self) -> None:
        self._tick_scheduled = False
        if not self._playing:
            return
        if self._duel is not None:
            self._duel_tick()
        elif self._instant_var.get():
            self._place_piece_instant()
        else:
            self._advance_animation()
        self._redraw_board()
        if self._playing:
            self._schedule_tick()

    def _advance_animation(self) -> None:
        env = self._env
        if self._anim is None:
            a = self._decide(self._policy, self._obs)
            if a < 0:
                self._finish_game()
                return
            rot, xi = divmod(int(a), config.BOARD_W)
            cells = pieces.piece_cells(env.cur_type, rot)
            c_min = min(c for (_, c) in cells)
            x = xi - c_min
            ty = env._drop_y(env.cur_type, rot, x, pieces.SPAWN_Y)
            self._anim = {'a': a, 'rot': rot, 'x': x,
                          'y': pieces.SPAWN_Y, 'ty': ty}
            return
        # падение на 1 клетку за тик
        self._anim['y'] += 1
        if self._anim['y'] >= self._anim['ty']:
            self._commit_placement(self._anim['a'])
            self._anim = None

    def _place_piece_instant(self) -> None:
        a = self._decide(self._policy, self._obs)
        if a < 0:
            self._finish_game()
            return
        self._commit_placement(a)

    def _commit_placement(self, a: int) -> None:
        self._obs, _r, done, info = self._env.step_placement(a)
        self._pieces += 1
        self._sv['lines'].set(str(info['score']))
        self._sv['pieces'].set(str(self._pieces))
        self._sv['next'].set(pieces.PIECE_NAMES[self._env.next_type])
        if done:
            self._finish_game()

    def _finish_game(self) -> None:
        self._games_write(
            f"#{self._game_no}: {self._env.score} lines, "
            f"{self._pieces} pcs")
        if self._restart_var.get() and self._playing:
            seed = self._pr_seed.get()
            self._env = TetrisEnv(
                seed=(seed if seed is not None else 0) + self._game_no,
                max_placements=0)
            self._obs = self._env.reset()
            self._anim = None
            self._pieces = 0
            self._reset_latency()
            self._game_no += 1
            self._sv['game'].set(str(self._game_no))
        else:
            self._playing = False
            self._btn_play.config(text="▶ Play")
            self._lbl_status.config(text="game over", fg=RED)

    # ── Drawing ──────────────────────────────────────────────────────────

    def _redraw_board(self) -> None:
        cv = self._canvas
        cv.delete("all")
        if self._duel is not None:
            d = self._duel
            off2 = config.BOARD_W * _CELL + self._DUEL_GAP
            self._draw_one(d['envs'][0], 0, anim=None, label="CNN")
            self._draw_one(d['envs'][1], off2, anim=None, label="EML")
            return
        self._draw_one(self._env, 0, anim=self._anim)

    def _draw_one(self, env: TetrisEnv, off: int, anim: dict | None,
                  label: str = "") -> None:
        cv = self._canvas
        W, H = config.BOARD_W, config.BOARD_H_TOTAL

        # Линия отделяет буфер спавна (верхние 4 строки) от видимой зоны.
        buf_y = config.BOARD_BUFFER * _CELL
        cv.create_rectangle(off, 0, off + W * _CELL, buf_y,
                            fill=BG2, outline="")
        cv.create_line(off, buf_y, off + W * _CELL, buf_y,
                       fill=RED, dash=(4, 2))

        # Сетка.
        for cx in range(W + 1):
            cv.create_line(off + cx * _CELL, 0, off + cx * _CELL, H * _CELL,
                           fill=BG2, width=1)
        for cy in range(H + 1):
            cv.create_line(off, cy * _CELL, off + W * _CELL, cy * _CELL,
                           fill=BG2, width=1)

        if label:
            color = ACCENT if label == "CNN" else GREEN
            cv.create_text(off + W * _CELL // 2, 12, text=f"{label}: "
                           f"{env.score} lines", fill=color,
                           font=("Consolas", 10, "bold"))

        # Зафиксированные ячейки.
        board = env.board
        for r in range(H):
            for c in range(W):
                if board[r, c]:
                    self._cell(off, c, r, ACCENT, outline=BG)

        if anim is not None:
            # Ghost на целевой позиции + падающая фигура.
            color = _PIECE_COLORS[env.cur_type]
            for (pr, pc) in pieces.piece_cells(env.cur_type, anim['rot']):
                self._cell(off, anim['x'] + pc, anim['ty'] + pr, "",
                           outline=color)
            for (pr, pc) in pieces.piece_cells(env.cur_type, anim['rot']):
                self._cell(off, anim['x'] + pc, anim['y'] + pr, color,
                           outline=BG)
        elif not env.done:
            # Фигура на спавне (ждёт решения).
            color = _PIECE_COLORS[env.cur_type]
            for (pr, pc) in pieces.piece_cells(env.cur_type, env.cur_rot):
                self._cell(off, env.cur_x + pc, env.cur_y + pr, color,
                           outline=BG)
        elif env.done:
            cv.create_text(off + W * _CELL // 2, H * _CELL // 2,
                           text="GAME OVER", fill=RED,
                           font=("Consolas", 13, "bold"))

    def _cell(self, off: int, c: int, r: int, fill: str, outline: str) -> None:
        self._canvas.create_rectangle(
            off + c * _CELL + 1, r * _CELL + 1,
            off + (c + 1) * _CELL - 1, (r + 1) * _CELL - 1,
            fill=fill, outline=outline, width=1)

    def _games_write(self, msg: str) -> None:
        self._games_log.config(state=tk.NORMAL)
        self._games_log.insert(tk.END, msg + "\n")
        self._games_log.see(tk.END)
        self._games_log.config(state=tk.DISABLED)


# ── Formulas tab ──────────────────────────────────────────────────────────────

def _check_formula_compat(trees: list) -> None:
    """Формулы старых версий (var_idx за пределами признаков v2) — отклонить."""
    def _max_var(node) -> int:
        if node.kind == 'var':
            return node.var_idx
        if node.kind == 'eml':
            return max(_max_var(node.left), _max_var(node.right))
        return 0
    mv = max(_max_var(t) for t in trees)
    if mv >= config.N_FEATURES:
        raise ValueError(
            f"формула несовместима: var_idx {mv} >= {config.N_FEATURES} "
            f"(старый формат v1 с 28 признаками)")


def _pretty_ast(node, indent: int = 0) -> str:
    pad = "  " * indent
    if node.kind in ('const', 'var'):
        return pad + node.to_string()
    return (pad + "eml(\n"
            + _pretty_ast(node.left, indent + 1) + ",\n"
            + _pretty_ast(node.right, indent + 1) + "\n"
            + pad + ")")


def _var_usage(node, counts: dict | None = None) -> dict:
    if counts is None:
        counts = {}
    if node.kind == 'var':
        name = (config.FEATURE_NAMES[node.var_idx]
                if node.var_idx < len(config.FEATURE_NAMES)
                else f"x{node.var_idx}")
        counts[name] = counts.get(name, 0) + 1
    elif node.kind == 'eml':
        _var_usage(node.left, counts)
        _var_usage(node.right, counts)
    return counts


class FormulasTab:
    """Вкладка Formulas: просмотр EML-формул и сравнение их игрой на одних сидах."""

    _POLL_MS = 250

    def __init__(self, root: tk.Tk, container: tk.Widget):
        self.root = root
        self.container = container
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._paths: list[str] = []

        self._build_ui()
        self._refresh_list()
        self._poll()

    def _build_ui(self) -> None:
        bar = tk.Frame(self.container, bg=BG)
        bar.pack(fill=tk.X, padx=10, pady=(6, 2))
        tk.Label(bar, text="Tetris AI  ·  EML Formulas",
                 bg=BG, fg=YELLOW, font=("Consolas", 12, "bold")).pack(side=tk.LEFT)
        self._lbl_status = tk.Label(bar, text="idle", bg=BG, fg=FG2,
                                    font=("Consolas", 9))
        self._lbl_status.pack(side=tk.RIGHT)

        cols = tk.Frame(self.container, bg=BG)
        cols.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # ── Left: список файлов + параметры сравнения ─────────────────────
        left = tk.Frame(cols, bg=BG2, width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)
        tk.Label(left, text="models/*.json  (multi-select)", bg=BG2, fg=YELLOW,
                 font=("Consolas", 8, "bold")).pack(pady=(8, 2))
        self._listbox = tk.Listbox(
            left, selectmode=tk.EXTENDED, bg=BG3, fg=FG,
            font=("Consolas", 8), relief=tk.FLAT, bd=0,
            selectbackground=ACCENT, selectforeground=BG, height=14)
        self._listbox.pack(fill=tk.BOTH, expand=False, padx=6, pady=2)
        self._listbox.bind("<<ListboxSelect>>", self._on_select)
        _btn(left, "Refresh", self._refresh_list, bg=BG3, fg=FG).pack(
            fill=tk.X, padx=6, pady=2)

        _sep(left)
        self._pr_games = ParamRow(left, "games/formula", 5, fmt=int)
        self._pr_maxpl = ParamRow(left, "max placements", 500, fmt=int)
        self._pr_seed = ParamRow(left, "seed", 1000, fmt=int)
        self._btn_cmp = _btn(left, "Compare selected", self._compare,
                             bg=YELLOW, fg=BG)
        self._btn_cmp.pack(fill=tk.X, padx=6, pady=6)
        tk.Label(left, text="одинаковые сиды у всех формул —\nчестное сравнение",
                 bg=BG2, fg=FG2, font=("Consolas", 7), justify="left").pack(padx=6)

        # ── Center: детали формулы ────────────────────────────────────────
        center = tk.Frame(cols, bg=BG)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(center, text="formula details", bg=BG, fg=FG2,
                 font=("Consolas", 8)).pack(anchor="w")
        self._details = tk.Text(center, bg=BG2, fg=FG, font=("Consolas", 9),
                                relief=tk.FLAT, bd=0, wrap=tk.NONE)
        dsb = tk.Scrollbar(center, command=self._details.yview)
        self._details.configure(yscrollcommand=dsb.set)
        dsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._details.pack(fill=tk.BOTH, expand=True)

        # ── Bottom: результаты сравнения ──────────────────────────────────
        bot = tk.Frame(self.container, bg=BG)
        bot.pack(fill=tk.X, padx=10, pady=(2, 8))
        tk.Label(bot, text="comparison results", bg=BG, fg=FG2,
                 font=("Consolas", 8)).pack(anchor="w")
        self._results = tk.Text(bot, height=8, bg=BG2, fg=FG2,
                                font=("Consolas", 9), state=tk.DISABLED,
                                relief=tk.FLAT, bd=0)
        self._results.pack(fill=tk.X)

    # ── Files ─────────────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        self._paths = sorted(glob.glob(os.path.join(config.MODELS_DIR, '*.json')))
        self._listbox.delete(0, tk.END)
        for p in self._paths:
            self._listbox.insert(tk.END, os.path.basename(p))

    def _on_select(self, _event=None) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        self._show_details(self._paths[sel[-1]])

    def _show_details(self, path: str) -> None:
        txt = [f"file: {path}", ""]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            meta = data.get('meta', {})
            trees = [EMLNode.from_dict(d) for d in data['trees']]
            for k, v in meta.items():
                txt.append(f"  {k}: {v}")
            txt.append("")
            try:
                _check_formula_compat(trees)
                compat = "v2 (19 afterstate-признаков) — играбельна"
            except ValueError as exc:
                compat = f"НЕСОВМЕСТИМА: {exc}"
            txt.append(f"  compat: {compat}")
            txt.append(f"  formulas: {len(trees)}")
            for i, t in enumerate(trees):
                txt.append("")
                txt.append(f"── formula {i}:  depth={t.depth()}  "
                           f"size={t.size()}  unique vars={t.n_unique_vars()}")
                usage = sorted(_var_usage(t).items(), key=lambda kv: -kv[1])
                txt.append("   vars: " + ", ".join(
                    f"{n}×{c}" for n, c in usage))
                txt.append(_pretty_ast(t, indent=1))
        except Exception as exc:  # noqa: BLE001 — показать в деталях
            txt.append(f"  [error] {exc!r}")
        self._details.delete("1.0", tk.END)
        self._details.insert("1.0", "\n".join(txt))

    # ── Compare ───────────────────────────────────────────────────────────

    def _compare(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        sel = self._listbox.curselection()
        if not sel:
            messagebox.showinfo("Compare", "Выберите одну или несколько формул.")
            return
        n_games = self._pr_games.get() or 5
        max_pl = self._pr_maxpl.get() or 500
        seed = self._pr_seed.get() or 1000
        paths = [self._paths[i] for i in sel]

        self._btn_cmp.config(state=tk.DISABLED)
        self._lbl_status.config(text="comparing…", fg=YELLOW)
        self._results_write(f"— comparing {len(paths)} formula(s), "
                            f"{n_games} games × {max_pl} placements, "
                            f"seed {seed}\n")

        def _run():
            rows = []
            for p in paths:
                name = os.path.basename(p)
                try:
                    trees, _ = storage.load_formulas(EMLNode, p)
                    _check_formula_compat(trees)
                    lines, steps = play_episodes(
                        trees[0], n_games=n_games,
                        max_placements=max_pl, seed=seed)
                    size = trees[0].size()
                    rows.append((lines, steps, size, name, None))
                    self._q.put(('row', (lines, steps, size, name, None)))
                except Exception as exc:  # noqa: BLE001
                    self._q.put(('row', (-1, 0, 0, name, repr(exc))))
            self._q.put(('done', rows))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        try:
            while True:
                kind, data = self._q.get_nowait()
                if kind == 'row':
                    lines, steps, size, name, err = data
                    if err:
                        self._results_write(f"  {name:<38} ERROR: {err}\n")
                    else:
                        self._results_write(
                            f"  {name:<38} {lines:>8.1f} lines  "
                            f"{steps:>6.0f} pcs  AST {size}\n")
                elif kind == 'done':
                    ok = [r for r in data if r[4] is None]
                    if ok:
                        best = max(ok)
                        self._results_write(
                            f"  BEST: {best[3]} ({best[0]:.1f} lines)\n")
                    self._btn_cmp.config(state=tk.NORMAL)
                    self._lbl_status.config(text="done", fg=FG2)
        except queue.Empty:
            pass
        self.root.after(self._POLL_MS, self._poll)

    def _results_write(self, msg: str) -> None:
        self._results.config(state=tk.NORMAL)
        self._results.insert(tk.END, msg)
        self._results.see(tk.END)
        self._results.config(state=tk.DISABLED)


# ── Module-level chart styler (accesses module constants) ────────────────────

def _style(ax) -> None:
    ax.tick_params(colors=FG2, labelsize=7)
    ax.grid(color=BG2, lw=0.5, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BG3)


# ── ttk Notebook styling (dark theme) ────────────────────────────────────────

def _style_notebook(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("default")
    except tk.TclError:
        pass
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=BG2, foreground=FG2,
                    font=("Consolas", 10, "bold"), padding=(18, 7), borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", BG3)],
              foreground=[("selected", ACCENT)])


# ── Entry point ───────────────────────────────────────────────────────────────

def main(initial_tab: str | None = None) -> None:
    root = tk.Tk()
    root.title("Tetris AI — Train · Distill · Play · Formulas")
    root.geometry("1240x800")
    root.configure(bg=BG)
    root.minsize(980, 620)

    _style_notebook(root)
    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True)

    frames = {}
    for key, text in [("train", "  Training  "),
                      ("distill", "  Distillation  "),
                      ("play", "  Play  "),
                      ("formulas", "  Formulas  ")]:
        frames[key] = tk.Frame(nb, bg=BG)
        nb.add(frames[key], text=text)

    TrainingGUI(root, frames["train"])
    DistillTab(root, frames["distill"])
    PlayTab(root, frames["play"])
    FormulasTab(root, frames["formulas"])

    if initial_tab in frames:
        nb.select(frames[initial_tab])

    root.mainloop()


if __name__ == "__main__":
    main()

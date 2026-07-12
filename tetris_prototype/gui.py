"""
gui.py — GUI монитор обучения CNN-оракула.

Запуск: python gui.py  или  python main.py gui

Панели:
  Left   — редактор гиперпараметров PPO (Apply применяет на лету во время обучения)
  Center — графики avg/max линий и потерь (pg/vf/entropy) + лог
  Right  — числовой статус в реальном времени
  Bottom — Start / Stop / Save Model
"""

import queue
import threading
import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
import storage
from cnn_oracle import TetrisCNN, get_device

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

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Tetris AI — Training Monitor")
        root.geometry("1200x740")
        root.configure(bg=BG)
        root.minsize(950, 580)

        self._q: queue.Queue       = queue.Queue()
        self._stop_evt             = threading.Event()
        self._thread: threading.Thread | None = None
        self._model: TetrisCNN | None         = None
        self._overrides: dict      = {}   # shared with training thread (GIL-safe reads)

        # Chart history
        self._upd:  list[int]   = []
        self._avg:  list[float] = []
        self._mx:   list[int]   = []
        self._pg:   list[float] = []
        self._vf:   list[float] = []
        self._ent:  list[float] = []

        self._build_ui()
        self._poll()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Title bar ────────────────────────────────────────────────────────
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill=tk.X, padx=10, pady=(6, 2))
        tk.Label(bar, text="Tetris AI  ·  Training Monitor",
                 bg=BG, fg=ACCENT, font=("Consolas", 12, "bold")).pack(side=tk.LEFT)
        self._lbl_status = tk.Label(bar, text="idle", bg=BG, fg=FG2,
                                    font=("Consolas", 9))
        self._lbl_status.pack(side=tk.RIGHT)

        # ── Main columns ─────────────────────────────────────────────────────
        cols = tk.Frame(self.root, bg=BG)
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
        bot = tk.Frame(self.root, bg=BG3)
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

        # Session params (need restart to apply)
        tk.Label(p, text="session (need restart to apply)",
                 bg=BG2, fg=FG2, font=("Consolas", 7)).pack()
        add("n_envs",    "n envs",      config.PPO_N_ENVS,      fmt=int)
        add("rollout",   "rollout",     config.PPO_ROLLOUT,      fmt=int)
        add("tot_steps", "total steps", config.PPO_TOTAL_STEPS,  fmt=int)

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

        _btn(p, "Clear Charts", self._clear_charts, bg=BG, fg=FG2).pack(
            side=tk.LEFT, padx=4, pady=7)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        hp = self._read_hp(full=True)
        if hp is None:
            return

        self._stop_evt.clear()
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._btn_save.config(state=tk.DISABLED)
        self._lbl_status.config(text="training", fg=GREEN)
        self._log_write("Training started.")

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
                overrides=self._overrides,
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
        messagebox.showinfo("Saved", f"Model saved:\n{path}")

    def _apply_overrides(self) -> None:
        hp = self._read_hp(full=False)
        if hp is None:
            return
        self._overrides.update(hp)
        self._log_write(
            f"Overrides applied — lr={hp['lr']:.2e}  clip={hp['clip']}  "
            f"ent={hp['ent_coef']}  vf={hp['vf_coef']}  "
            f"ep={hp['epochs']}  bs={hp['batch_size']}"
        )

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
                elif kind == "done":
                    self._on_done(data)
        except queue.Empty:
            pass
        if dirty:
            self._redraw()
        self.root.after(self._POLL_MS, self._poll)

    def _absorb(self, rec: dict) -> None:
        """Принять один update-record из очереди."""
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


# ── Module-level chart styler (accesses module constants) ────────────────────

def _style(ax) -> None:
    ax.tick_params(colors=FG2, labelsize=7)
    ax.grid(color=BG2, lw=0.5, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BG3)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""
gui.py — project interface: data · features · training · distillation · test.

The style and helpers are taken from tetris/gui.py so the sub-projects look
alike.

Tabs:
    Data          — dataset composition, sample crops of every class.
    Features      — F-statistics and per-class feature distributions.
    Training      — oracle training, live loss / val accuracy curves.
    Distillation  — GA run, live curves of both phases, formula text.
    Test          — two inner tabs: Photo and Video.

Every plot can be saved with the "Save plot" button — the file goes into
results/ under a descriptive name.

Three decisions worth explaining:

1. Heavy operations (training, distillation, video decoding) run in separate
   threads, and the results are passed to the interface through a queue.Queue
   and consumed in the main thread via root.after(). tkinter is not thread-safe:
   drawing from a worker thread means rare, irreproducible crashes.

   Progress goes through the same channel: training and distillation accept
   progress_cb and should_stop, so the curves are drawn as the work proceeds,
   and "Stop" halts at the nearest epoch/generation while keeping the best
   result found. Previously both tasks stayed silent until the very end — for
   distillation that meant tens of minutes of a frozen window.

2. Frames are displayed through tk.Canvas and PhotoImage from a PPM assembled by
   hand from numpy. Image display therefore requires neither Pillow nor OpenCV —
   so photo mode works even where the video tab is disabled.

3. Only one heavy task can run in the application (BusyLock). Training rewrites
   models/oracle_X.pt and distillation reads it — running them together means
   reading the checkpoint while it is being written.
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
import vision


# ── Catppuccin Mocha palette (as in tetris/gui.py) ───────────────────────────
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

MONO = ("Consolas", 9)
MONO_B = ("Consolas", 9, "bold")


# ── Widget helpers ────────────────────────────────────────────────────────────

def _sep(parent: tk.Widget) -> None:
    tk.Frame(parent, bg=BG3, height=1).pack(fill=tk.X, padx=4, pady=3)


def _btn(parent: tk.Widget, text: str, cmd, *, bg: str = BG3, fg: str = FG,
         state=tk.NORMAL, side=tk.TOP, **pack) -> tk.Button:
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                  activebackground=fg, activeforeground=bg, font=MONO_B,
                  relief=tk.FLAT, bd=0, padx=12, pady=6, state=state,
                  cursor="hand2")
    b.pack(side=side, **pack)
    return b


def _label(parent: tk.Widget, text: str, *, fg: str = FG2, font=MONO,
           **pack) -> tk.Label:
    lbl = tk.Label(parent, text=text, bg=parent['bg'], fg=fg, font=font,
                   anchor='w', justify=tk.LEFT)
    lbl.pack(**pack)
    return lbl


def _section(parent: tk.Widget, title: str) -> tk.Frame:
    """A titled block of the control panel."""
    tk.Label(parent, text=title, bg=parent['bg'], fg=ACCENT,
             font=("Consolas", 9, "bold"), anchor='w').pack(fill=tk.X,
                                                            padx=6, pady=(8, 2))
    frame = tk.Frame(parent, bg=parent['bg'])
    frame.pack(fill=tk.X)
    return frame


class ParamRow:
    """A "Label + Entry" row for a single parameter."""

    def __init__(self, parent: tk.Widget, label: str, default, fmt=float):
        self._fmt = fmt
        row = tk.Frame(parent, bg=parent['bg'])
        row.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(row, text=label, bg=parent['bg'], fg=FG2, font=("Consolas", 8),
                 width=14, anchor="w").pack(side=tk.LEFT)
        self._var = tk.StringVar(value=str(default))
        tk.Entry(row, textvariable=self._var, bg=BG3, fg=FG,
                 insertbackground=FG, font=("Consolas", 8), width=10,
                 relief=tk.FLAT, bd=2).pack(side=tk.LEFT, padx=(2, 0))

    def get(self, fallback=None):
        try:
            return self._fmt(self._var.get())
        except (ValueError, TypeError):
            return fallback

    def set(self, value):
        self._var.set(str(value))


# ── Displaying images through PPM ─────────────────────────────────────────────

def _to_photo(rgb: np.ndarray) -> tk.PhotoImage:
    """
    numpy (H, W, 3) uint8 -> tk.PhotoImage through the PPM format.

    Tk reads PPM out of the box, so Pillow is not needed: the header and raw
    bytes are assembled by hand.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    header = f"P6\n{w} {h}\n255\n".encode('ascii')
    return tk.PhotoImage(data=header + rgb.tobytes())


def _drain(q: queue.Queue) -> None:
    """Empty a queue — so that a source switch does not show foreign frames."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _color_legend(parent: tk.Widget) -> tk.Frame:
    """A "color — class" strip: without it the overlay reads as random blobs."""
    row = tk.Frame(parent, bg=parent['bg'])
    tk.Label(row, text="classes:", bg=parent['bg'], fg=FG2,
             font=("Consolas", 8)).pack(side=tk.LEFT, padx=(0, 6))
    for i, name in enumerate(config.CLASS_NAMES):
        color = '#%02x%02x%02x' % vision.class_color(i)
        tk.Label(row, text="  ", bg=color).pack(side=tk.LEFT)
        tk.Label(row, text=f" {name}  ", bg=parent['bg'], fg=FG,
                 font=("Consolas", 8)).pack(side=tk.LEFT)
    return row


class ImageView(tk.Canvas):
    """
    Canvas for showing a frame with labels on top of it.

    The labels are drawn as canvas items rather than burned into the pixels: the
    text stays sharp at any scale and does not corrupt the saved frame.
    """

    def __init__(self, parent, width=640, height=480):
        super().__init__(parent, bg=BG, highlightthickness=0,
                         width=width, height=height)
        self._photo = None          # keep the reference, otherwise Tk collects it

    def show(self, rgb: np.ndarray, labels=None):
        """
        Show a frame. labels — a list of (x, y, text, color) in frame pixels.
        """
        self.delete('all')
        self._photo = _to_photo(rgb)
        h, w = rgb.shape[:2]
        self.config(width=w, height=h)
        self.create_image(0, 0, anchor=tk.NW, image=self._photo)
        for x, y, text, color in (labels or []):
            self.create_text(x + 3, y + 3, text=text, anchor=tk.NW,
                             fill='#000000', font=("Consolas", 8, "bold"))
            self.create_text(x + 2, y + 2, text=text, anchor=tk.NW,
                             fill=color, font=("Consolas", 8, "bold"))


# ── Panel with a plot and a save button ───────────────────────────────────────

class PlotPanel(tk.Frame):
    """
    A matplotlib figure inside the window plus a "Save plot" button.

    The dark theme is set here once, so that every plot does not configure the
    colors again and all tabs look the same.
    """

    def __init__(self, parent, name: str, figsize=(7.5, 4.5)):
        super().__init__(parent, bg=BG)
        self.name = name
        self.fig = Figure(figsize=figsize, dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill=tk.X, pady=(2, 0))
        _btn(bar, "Save plot", self.save, side=tk.RIGHT, padx=4)
        self._status = _label(bar, "", fg=FG2, side=tk.RIGHT, padx=6)

    def axes(self, *args, **kwargs):
        """Create axes already styled for the dark theme."""
        ax = self.fig.add_subplot(*args, **kwargs)
        ax.set_facecolor(BG2)
        for spine in ax.spines.values():
            spine.set_color(BG3)
        ax.tick_params(colors=FG2, labelsize=8)
        ax.xaxis.label.set_color(FG2)
        ax.yaxis.label.set_color(FG2)
        ax.title.set_color(FG)
        ax.grid(alpha=0.15, color=FG2)
        return ax

    def clear(self):
        self.fig.clear()

    def draw(self):
        self.canvas.draw_idle()

    def save(self):
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR, f"{self.name}.png")
        self.fig.savefig(path, dpi=150, facecolor=BG)
        self._status.config(text=f"-> {path}")


# ── Application-wide busy state ───────────────────────────────────────────────

class BusyLock:
    """
    One heavy task for the whole application.

    Training rewrites models/oracle_X.pt, distillation reads the same file and
    writes models/best_eml_X.json, and both touch shared caches. Nothing
    prevented starting them simultaneously — and distilling an oracle that was
    being overwritten at that moment. The tab that starts work therefore takes
    the lock, and the others disable their start buttons for the duration.

    The listeners are called ONLY from the main thread (acquire/release are
    invoked from button handlers and from queue processing), otherwise they
    would touch widgets from a foreign thread.
    """

    def __init__(self):
        self._owner: str | None = None
        self._listeners: list = []

    @property
    def owner(self) -> str | None:
        return self._owner

    def subscribe(self, callback):
        """callback(owner | None) — called on owner change and immediately."""
        self._listeners.append(callback)
        callback(self._owner)

    def acquire(self, name: str) -> bool:
        if self._owner is not None:
            messagebox.showinfo(
                "Busy", f"Currently running: {self._owner}.\n"
                        f"Wait for it to finish or press \"Stop\".")
            return False
        self._owner = name
        self._notify()
        return True

    def release(self, name: str):
        if self._owner == name:
            self._owner = None
            self._notify()

    def _notify(self):
        for callback in self._listeners:
            callback(self._owner)


# ── Common ancestor of tabs with a background task ────────────────────────────

class WorkerTab:
    """
    A tab that starts a long operation in a separate thread.

    The thread puts events into a queue and the main thread consumes it on a
    timer — drawing from a worker thread is not allowed, tkinter does not
    survive it.
    """

    def __init__(self, root: tk.Tk, lock: BusyLock | None = None,
                 lock_name: str | None = None):
        self.root = root
        self.queue: queue.Queue = queue.Queue()
        self.thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self.lock = lock
        self.lock_name = lock_name
        self._poll()

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, target, *args, **kwargs):
        if self.busy:
            messagebox.showinfo("Busy", "The operation is already running.")
            return False
        if self.lock is not None and not self.lock.acquire(self.lock_name):
            return False
        self._stop_flag.clear()
        self.thread = threading.Thread(target=self._wrap, args=(target, args, kwargs),
                                       daemon=True)
        self.thread.start()
        return True

    def _wrap(self, target, args, kwargs):
        try:
            target(*args, **kwargs)
        except Exception as exc:                      # noqa: BLE001
            import sys
            import traceback
            trace = traceback.format_exc()
            # The window shows only a short message, while the full stack goes
            # to the console: without it, debugging a background task turns into
            # guessing from a single line.
            print(trace, file=sys.stderr)
            self.queue.put({'type': 'error', 'text': f"{exc}", 'trace': trace})
        finally:
            self.queue.put({'type': 'done'})

    def request_stop(self):
        self._stop_flag.set()

    def should_stop(self) -> bool:
        return self._stop_flag.is_set()

    def _poll(self):
        try:
            while True:
                event = self.queue.get_nowait()
                # The lock is released here rather than in the worker thread:
                # its listeners touch widgets, which is only allowed from here.
                if event.get('type') == 'done' and self.lock is not None:
                    self.lock.release(self.lock_name)
                self.on_event(event)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def on_event(self, event: dict):
        """Overridden by subclasses."""


# ── Tab: Data ─────────────────────────────────────────────────────────────────

class DataTab:
    """Dataset composition and sample crops of every class."""

    def __init__(self, root: tk.Tk, parent: tk.Frame):
        self.root = root
        self.images = self.labels = None

        panel = tk.Frame(parent, bg=BG2, width=210)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "CLASS FOLDER")
        _btn(panel, "Choose folder...", self.choose_folder, bg=PURPLE, fg=BG,
             fill=tk.X, padx=6, pady=3)
        self.path_info = _label(panel, "", fill=tk.X, padx=8, pady=(2, 4))
        _sep(panel)

        _section(panel, "DATASET")
        _btn(panel, "Load dataset", self.load, fill=tk.X, padx=6, pady=3)
        _btn(panel, "Other samples", self.resample, fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "dataset not loaded", fill=tk.X, padx=8,
                           pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.plot = PlotPanel(right, "gui_dataset_distribution", figsize=(7, 2.6))
        self.plot.pack(fill=tk.X, padx=6, pady=6)

        _label(right, "sample crops (one row per class)", fg=ACCENT,
               font=MONO_B, fill=tk.X, padx=8)
        self.view = ImageView(right, width=880, height=300)
        self.view.pack(padx=8, pady=6)
        self._show_path()

    def _show_path(self):
        """
        Which folder is currently selected — visible before any loading.

        The window title is updated here too. It is set at startup, and without
        this it would keep showing the PREVIOUS class set after a dataset
        change — i.e. it would lie about exactly the thing every other number on
        screen depends on.
        """
        self.path_info.config(
            text=f"{os.path.basename(os.path.abspath(config.DATASET_DIR))}\n"
                 f"{config.N_CLASSES} cls.: " + ', '.join(config.CLASS_NAMES))
        self.root.title(f"EML Classifier — {config.dataset_summary()}")

    def choose_folder(self):
        """
        Choose a folder whose subdirectories become the classes.

        The folder contents are shown BEFORE applying: a dataset change changes
        the number of classes and therefore makes already trained models
        incompatible. Learning that from a dialog is better than from a refusal
        to load the oracle.
        """
        import data

        path = filedialog.askdirectory(
            title="Folder with class subdirectories", mustexist=True,
            initialdir=os.path.abspath(config.DATASET_DIR)
                       if os.path.isdir(config.DATASET_DIR) else None)
        if not path:
            return

        info = data.describe_root(path)
        if not info['classes']:
            messagebox.showerror(
                "Not suitable",
                f"There are no subdirectories with images in {path}.\n\n"
                f"Expected:\n  <folder>/CLASS_NAME/*.bmp|png|jpg\n\n"
                f"Every subdirectory is a class, its name is the class name.")
            return

        listing = '\n'.join(f"    {name:<20} {info['counts'][name]:>6}"
                            for name in info['classes'])
        same = info['classes'] == list(config.CLASS_NAMES)
        warning = "" if same else (
            "\n\nTHE CLASS SET IS DIFFERENT. Already trained oracles and\n"
            "formulas will not apply to it — they must be retrained.")

        # JPEG is not read by the own decoder. Saying so here is cheaper than
        # failing dataset loading five minutes after the choice.
        import imaging
        if ({'.jpg', '.jpeg'} & set(info['exts'])
                and imaging.backend_name() == 'none'):
            warning += ("\n\nThe folder contains JPEG but there is no decoder "
                        "for it.\nInstall opencv-python or Pillow — otherwise "
                        "dataset\nloading will stop at the first .jpg.")
        if not messagebox.askokcancel(
                "Use this dataset?",
                f"{path}\n\nclasses: {len(info['classes'])}, "
                f"images: {info['n_files']}\n"
                f"formats: {', '.join(info['exts'])}\n"
                f"sizes: {', '.join(info['sizes'])}"
                f" (converted to {config.IMG_SIZE}x{config.IMG_SIZE})\n\n"
                f"{listing}{warning}"):
            return

        config.use_dataset(path, info['classes'])
        config.save_dataset_choice()
        self.images = self.labels = None
        self._show_path()
        self.info.config(text="dataset selected,\npress \"Load dataset\"")
        self.plot.clear()
        self.plot.draw()

    def load(self):
        import data
        self.images, self.labels, _ = data.load_dataset(verbose=False)
        counts = np.bincount(self.labels, minlength=config.N_CLASSES)
        train, val, test = data.stratified_split(self.labels)
        self.info.config(
            text=f"total: {len(self.labels)}\n"
                 f"train: {len(train)}\nval:   {len(val)}\ntest:  {len(test)}\n\n"
                 + "\n".join(f"{c:<7} {counts[i]:>5}"
                             for i, c in enumerate(config.CLASS_NAMES))
                 + f"\n\nimbalance {counts.max() / counts.min():.1f} : 1")
        self._draw_distribution(counts)
        self.resample()

    def _draw_distribution(self, counts):
        self.plot.clear()
        ax = self.plot.axes(111)
        colors = [np.array(vision.class_color(i)) / 255 for i in range(config.N_CLASSES)]
        ax.bar(config.CLASS_NAMES, counts, color=colors)
        for i, v in enumerate(counts):
            ax.text(i, v, str(v), ha='center', va='bottom', color=FG, fontsize=8)
        ax.set_title("images per class")
        self.plot.draw()

    def resample(self):
        if self.images is None:
            return
        rng = np.random.default_rng()
        per_row = 12
        rows = []
        for c in range(config.N_CLASSES):
            idx = np.flatnonzero(self.labels == c)
            sel = rng.choice(idx, min(per_row, len(idx)), replace=False)
            rows.append(np.concatenate(list(self.images[sel]), axis=1))
        self.view.show(np.concatenate(rows, axis=0),
                       labels=[(4, c * config.IMG_SIZE + 4, config.CLASS_NAMES[c],
                                '#%02x%02x%02x' % vision.class_color(c))
                               for c in range(config.N_CLASSES)])


# ── Tab: Features ─────────────────────────────────────────────────────────────

class FeaturesTab:
    """Feature F-statistics and their per-class distributions."""

    def __init__(self, root: tk.Tk, parent: tk.Frame):
        self.root = root
        self.feats = self.labels = None

        panel = tk.Frame(parent, bg=BG2, width=210)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "FEATURES")
        _btn(panel, "Compute features", self.compute, fill=tk.X, padx=6, pady=3)
        _sep(panel)
        _label(panel, "feature to inspect:", fill=tk.X, padx=8)
        self.choice = tk.StringVar(value=config.FEATURE_NAMES[0])
        self.combo = ttk.Combobox(panel, textvariable=self.choice, font=("Consolas", 8),
                                  values=config.FEATURE_NAMES, state='readonly')
        self.combo.pack(fill=tk.X, padx=6, pady=4)
        self.combo.bind('<<ComboboxSelected>>', lambda e: self._draw_detail())
        _sep(panel)
        self.info = _label(panel, "not computed", fill=tk.X, padx=8, pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot_f = PlotPanel(right, "gui_feature_fscore", figsize=(8, 3.4))
        self.plot_f.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.plot_d = PlotPanel(right, "gui_feature_detail", figsize=(8, 2.8))
        self.plot_d.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

    def compute(self):
        import data
        import features as features_mod

        images, self.labels, _ = data.load_dataset(verbose=False)
        self.feats = features_mod.load_or_compute(images, verbose=False)
        train, _, _ = data.stratified_split(self.labels)
        self.fscore = features_mod.anova_f(self.feats[train], self.labels[train])
        order = np.argsort(-self.fscore)
        self.info.config(
            text=f"features: {config.N_FEATURES}\n\nstrongest:\n"
                 + "\n".join(f"{config.FEATURE_NAMES[i][:15]:<15}{self.fscore[i]:6.0f}"
                             for i in order[:6])
                 + "\n\nweakest:\n"
                 + "\n".join(f"{config.FEATURE_NAMES[i][:15]:<15}{self.fscore[i]:6.0f}"
                             for i in order[-3:]))
        self._draw_fscore(order)
        self.choice.set(config.FEATURE_NAMES[order[0]])
        self._draw_detail()

    def _draw_fscore(self, order):
        self.plot_f.clear()
        ax = self.plot_f.axes(111)
        names = [config.FEATURE_NAMES[i] for i in order]
        ax.barh(range(len(order)), self.fscore[order], color=ACCENT)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.set_title("F-statistic: how well a feature separates the classes")
        self.plot_f.fig.tight_layout()
        self.plot_f.draw()

    def _draw_detail(self):
        if self.feats is None:
            return
        idx = config.FEATURE_NAMES.index(self.choice.get())
        self.plot_d.clear()
        ax = self.plot_d.axes(111)
        for c in range(config.N_CLASSES):
            vals = self.feats[self.labels == c, idx]
            ax.hist(vals, bins=40, alpha=0.55, label=config.CLASS_NAMES[c],
                    color=np.array(vision.class_color(c)) / 255)
        ax.set_title(f"{self.choice.get()}  (F = {self.fscore[idx]:.0f})")
        leg = ax.legend(fontsize=7, facecolor=BG2, edgecolor=BG3)
        for text in leg.get_texts():
            text.set_color(FG2)
        self.plot_d.fig.tight_layout()
        self.plot_d.draw()


# ── Tab: Training ─────────────────────────────────────────────────────────────

class TrainTab(WorkerTab):
    """Oracle training with live curves."""

    REDRAW_INTERVAL = 0.4        # s; redrawing the curves more often is pointless

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        super().__init__(root, lock=lock, lock_name="oracle training")
        self.history = {'train_loss': [], 'val_acc': [], 'val_bal_acc': []}
        self._last_draw = 0.0

        panel = tk.Frame(parent, bg=BG2, width=210)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "BRANCH")
        self.branch = tk.StringVar(value='A')
        for value, text in (('A', "A — MLP over features"),
                            ('B', "B — CNN over pixels")):
            tk.Radiobutton(panel, text=text, variable=self.branch, value=value,
                           bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                           activebackground=BG2, activeforeground=ACCENT,
                           anchor='w').pack(fill=tk.X, padx=6)

        params = _section(panel, "PARAMETERS")
        self.p_epochs = ParamRow(params, "epochs", config.TRAIN_EPOCHS, int)
        self.p_lr = ParamRow(params, "LR", config.TRAIN_LR, float)
        self.p_batch = ParamRow(params, "batch", config.TRAIN_BATCH_SIZE, int)
        # Name of the library copy. Empty = time plus metric.
        self.p_name = ParamRow(params, "model name", "", str)

        _sep(panel)
        self.btn = _btn(panel, "Train", self.run, bg=GREEN, fg=BG,
                        fill=tk.X, padx=6, pady=3)
        self.btn_stop = _btn(panel, "Stop", self.stop, bg=RED, fg=BG,
                             state=tk.DISABLED, fill=tk.X, padx=6, pady=3)
        self.btn_saved = _btn(panel, "Show saved", self.load_saved,
                              fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "ready", fill=tk.X, padx=8, pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot = PlotPanel(right, "gui_training", figsize=(8, 5.5))
        self.plot.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        if lock is not None:
            lock.subscribe(self._on_lock)

    def _on_lock(self, owner):
        """While another tab is working, starting from here is unavailable."""
        foreign = owner is not None and owner != self.lock_name
        state = tk.DISABLED if foreign else tk.NORMAL
        self.btn_saved.config(state=state)
        if foreign:
            self.btn.config(state=tk.DISABLED)
        elif owner is None:
            self.btn.config(state=tk.NORMAL, text="Train")

    def run(self):
        branch = self.branch.get()
        epochs = self.p_epochs.get(config.TRAIN_EPOCHS)
        # LR and batch travel as run PARAMETERS. Previously they were written
        # into config.TRAIN_LR / config.TRAIN_BATCH_SIZE and stayed there until
        # the end of the session, changing the behavior of neighboring tabs and
        # of subsequent runs.
        lr = self.p_lr.get(config.TRAIN_LR)
        batch = self.p_batch.get(config.TRAIN_BATCH_SIZE)
        name = (self.p_name.get('') or '').strip() or None
        self.history = {'train_loss': [], 'val_acc': [], 'val_bal_acc': []}
        self._last_draw = 0.0
        if self.start(self._work, branch, epochs, lr, batch, name):
            self.btn.config(state=tk.DISABLED, text="Training...")
            self.btn_stop.config(state=tk.NORMAL)
            self.info.config(text=f"branch {branch}: training...")

    def stop(self):
        self.request_stop()
        self.btn_stop.config(state=tk.DISABLED)
        self.info.config(text="stopping after the epoch...")

    def _work(self, branch, epochs, lr, batch, name=None):
        import train_oracle

        def progress(payload):
            self.queue.put({'type': 'epoch', **payload})

        out = train_oracle.train(branch, epochs=epochs, verbose=False,
                                 lr=lr, batch_size=batch, name=name,
                                 progress_cb=progress,
                                 should_stop=self.should_stop)
        self.queue.put({'type': 'trained', 'history': out['history'],
                        'meta': out['meta']})

    def load_saved(self):
        import storage
        meta = storage.oracle_meta(self.branch.get())
        if not meta:
            messagebox.showinfo("No model",
                                f"The branch {self.branch.get()} oracle is not "
                                f"trained.")
            return
        self.on_event({'type': 'trained', 'history': meta.get('history', {}),
                       'meta': meta})

    def on_event(self, event):
        if event['type'] == 'epoch':
            self.history = event['history']
            self.info.config(
                text=f"epoch {event['epoch'] + 1}/{event['epochs']}\n"
                     f"loss: {event['train_loss']:.4f}\n"
                     f"val bal: {event['val_bal_acc']:.4f}\n"
                     f"val acc: {event['val_acc']:.4f}\n"
                     f"best:    {event['best_val_bal_acc']:.4f}\n"
                     f"LR: {event['lr']:.2e}")
            now = time.perf_counter()
            if now - self._last_draw > self.REDRAW_INTERVAL:
                self._last_draw = now
                self._draw()
        elif event['type'] == 'trained':
            self.history = event.get('history') or {}
            meta = event['meta']
            note = "\n(stopped manually)" if meta.get('stopped_early') else ""
            self.info.config(
                text=f"{meta.get('model', '?')}\n"
                     f"parameters: {meta.get('params', 0):,}\n"
                     f"val bal:  {meta.get('val_bal_acc', 0):.4f}\n"
                     f"val acc:  {meta.get('val_acc', 0):.4f}\n"
                     f"test bal: {meta.get('test_bal_acc', 0):.4f}\n"
                     f"test acc: {meta.get('test_acc', 0):.4f}\n"
                     f"epochs: {meta.get('epochs_run', 0)}\n"
                     f"time: {meta.get('seconds', 0):.0f} s{note}\n\n"
                     f"in the library:\n{meta.get('library_name', '—')}")
            self._draw()
        elif event['type'] == 'error':
            messagebox.showerror("Training error", event['text'])
            self.info.config(text="error")
        elif event['type'] == 'done':
            self.btn.config(state=tk.NORMAL, text="Train")
            self.btn_stop.config(state=tk.DISABLED)

    def _draw(self):
        loss = self.history.get('train_loss') or []
        acc = self.history.get('val_acc') or []
        bal = self.history.get('val_bal_acc') or []
        if not loss:
            return
        self.plot.clear()
        ax1 = self.plot.axes(211)
        ax1.plot(loss, color=RED)
        ax1.set_title("train loss")
        ax2 = self.plot.axes(212)
        # Both curves: selection uses the balanced one, and the gap between them
        # shows how much the model leans on the frequent class.
        ax2.plot(acc, color=FG2, alpha=0.8, label="val accuracy")
        curve = bal or acc
        if bal:
            ax2.plot(bal, color=GREEN, label="val bal. accuracy")
        best = int(np.argmax(curve))
        ax2.axvline(best, color=YELLOW, ls='--', alpha=0.7)
        ax2.set_title(f"{'val bal.accuracy' if bal else 'val accuracy'} "
                      f"(best {curve[best]:.4f} at epoch {best})")
        ax2.set_xlabel("epoch")
        leg = ax2.legend(fontsize=7, facecolor=BG2, edgecolor=BG3)
        for text in leg.get_texts():
            text.set_color(FG2)
        self.plot.fig.tight_layout()
        self.plot.draw()


# ── Tab: Distillation ─────────────────────────────────────────────────────────

class DistillTab(WorkerTab):
    """GA distillation with live curves of both phases."""

    REDRAW_INTERVAL = 0.5        # s; GA curves twitch every generation

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        super().__init__(root, lock=lock, lock_name="distillation")
        self.data_hist = {}
        self.acc_hist = []
        self.single_hist = []
        self.island_hist = []
        self._last_draw = 0.0

        panel = tk.Frame(parent, bg=BG2, width=215)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "BRANCH")
        self.branch = tk.StringVar(value='A')
        for value, text in (('A', "A — one formula per class"),
                            ('B', "B — CNN embedding"),
                            ('C', "C — ONE formula")):
            tk.Radiobutton(panel, text=text, variable=self.branch, value=value,
                           bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                           activebackground=BG2, activeforeground=ACCENT,
                           anchor='w', command=self._on_branch
                           ).pack(fill=tk.X, padx=6)

        # The defaults come from config rather than being hard-coded numbers:
        # previously 300/400 stood here against 600/600 in config, and a run
        # from the window silently computed something other than
        # `python main.py distill` — while their results were then compared.
        params = _section(panel, "EVOLUTION")
        self.p_gen = ParamRow(params, "generations", config.EML_GENERATIONS, int)
        self.p_pop = ParamRow(params, "population", config.EML_POPULATION, int)
        self.p_seed = ParamRow(params, "seed", 0, int)
        self.p_name = ParamRow(params, "model name", "", str)
        _label(panel, "complexity penalty:", fill=tk.X, padx=8, pady=(6, 0))
        self.penalty = tk.StringVar(value=config.DEPTH_PENALTY_DEFAULT)
        ttk.Combobox(panel, textvariable=self.penalty, font=("Consolas", 8),
                     values=list(config.DEPTH_PENALTIES), state='readonly'
                     ).pack(fill=tk.X, padx=6, pady=3)

        self.acc_phase = tk.BooleanVar(value=True)
        self.chk_acc = tk.Checkbutton(
            panel, text="ACC phase", variable=self.acc_phase, bg=BG2, fg=FG,
            selectcolor=BG3, font=("Consolas", 8), activebackground=BG2,
            anchor='w')
        self.chk_acc.pack(fill=tk.X, padx=6, pady=4)

        # Branch C runs for hours, so its controls are about how to spend that
        # time: how many independent populations and how many processes for them.
        self.c_frame = tk.Frame(panel, bg=BG2)
        self.p_islands = ParamRow(self.c_frame, "islands", 0, int)
        self.p_workers = ParamRow(self.c_frame, "processes", 0, int)
        self.resume = tk.BooleanVar(value=True)
        tk.Checkbutton(self.c_frame, text="resume the run",
                       variable=self.resume, bg=BG2, fg=FG, selectcolor=BG3,
                       font=("Consolas", 8), activebackground=BG2,
                       anchor='w').pack(fill=tk.X, padx=6, pady=2)
        _label(self.c_frame, "0 = one per core", fg=FG2, fill=tk.X, padx=8)
        _sep(panel)
        self.btn = _btn(panel, "Distill", self.run, bg=PURPLE, fg=BG,
                        fill=tk.X, padx=6, pady=3)
        self.btn_stop = _btn(panel, "Stop", self.stop, bg=RED, fg=BG,
                             state=tk.DISABLED, fill=tk.X, padx=6, pady=3)
        self.btn_saved = _btn(panel, "Show saved", self.load_saved,
                              fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "ready", fill=tk.X, padx=8, pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot = PlotPanel(right, "gui_distill", figsize=(8, 3.6))
        self.plot.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        _label(right, "formulas", fg=ACCENT, font=MONO_B, fill=tk.X, padx=8)
        self.text = tk.Text(right, bg=BG2, fg=FG, font=("Consolas", 8),
                            relief=tk.FLAT, height=11, wrap=tk.WORD,
                            insertbackground=FG)
        self.text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))

        if lock is not None:
            lock.subscribe(self._on_lock)
        self._on_branch()

    def _on_branch(self):
        """
        Show the controls of the selected branch.

        Branch C has no ACC phase (it is one phase in total), but it does have
        islands, processes and run resumption. Leaving foreign fields on screen
        would offer knobs that affect nothing.
        """
        if self.branch.get() == 'C':
            self.chk_acc.pack_forget()
            self.c_frame.pack(fill=tk.X)
            self.p_gen.set(config.EMLC_GENERATIONS)
            self.p_pop.set(config.EMLC_POPULATION)
        else:
            self.c_frame.pack_forget()
            self.chk_acc.pack(fill=tk.X, padx=6, pady=4)
            self.p_gen.set(config.EML_GENERATIONS)
            self.p_pop.set(config.EML_POPULATION)

    def _on_lock(self, owner):
        foreign = owner is not None and owner != self.lock_name
        state = tk.DISABLED if foreign else tk.NORMAL
        self.btn_saved.config(state=state)
        if foreign:
            self.btn.config(state=tk.DISABLED)
        elif owner is None:
            self.btn.config(state=tk.NORMAL, text="Distill")

    def run(self):
        branch = self.branch.get()
        defaults = ((config.EMLC_GENERATIONS, config.EMLC_POPULATION)
                    if branch == 'C'
                    else (config.EML_GENERATIONS, config.EML_POPULATION))
        if not self.start(self._work, branch,
                          self.p_gen.get(defaults[0]),
                          self.p_pop.get(defaults[1]),
                          self.penalty.get(), self.acc_phase.get(),
                          self.p_seed.get(0),
                          (self.p_name.get('') or '').strip() or None):
            return
        self.data_hist, self.acc_hist = {}, []
        self.single_hist, self.island_hist = [], []
        self._last_draw = 0.0
        self.btn.config(state=tk.DISABLED, text="Evolving...")
        self.btn_stop.config(state=tk.NORMAL)
        self.info.config(text="distilling...")
        self.text.delete('1.0', tk.END)

    def stop(self):
        self.request_stop()
        self.btn_stop.config(state=tk.DISABLED)
        self.info.config(text="stopping after the generation...")

    def _work(self, branch, generations, population, penalty, acc_phase, seed,
              name=None):
        import pipeline

        # Progress arrives every generation — thousands of events per run. We
        # enqueue at most once per 0.2 s: the interface needs no more, and there
        # is no point filling the queue with history lists.
        last = [0.0]

        def progress(payload):
            now = time.perf_counter()
            final = payload['gen'] == payload['generations'] - 1
            if final or now - last[0] > 0.2:
                last[0] = now
                self.queue.put({'type': 'progress', **payload})

        if branch == 'C':
            # Branch C computes in separate PROCESSES (see eml_single). Doing so
            # from the interface thread is legitimate: child processes are
            # created via spawn, and the project entry point is guarded by a
            # __main__ check.
            out = pipeline.run_single(
                'C', size_penalty=penalty, generations=generations,
                population=population, islands=self.p_islands.get(0) or None,
                workers=self.p_workers.get(0) or None, seed=seed,
                resume=self.resume.get(), save=True, verbose=False, name=name,
                progress_cb=progress, should_stop=self.should_stop)
        else:
            out = pipeline.run(branch, depth_penalty=penalty,
                               generations=generations, population=population,
                               acc_phase=acc_phase, save=True, verbose=False,
                               seed=seed, name=name, progress_cb=progress,
                               should_stop=self.should_stop)
        self.queue.put({'type': 'distilled', 'meta': out['meta'],
                        'result': out['result'], 'saved': out.get('saved'),
                        'branch': branch,
                        'formulas': [t.to_string() for t in out['trees']]})

    def load_saved(self):
        import storage
        try:
            clf, payload = storage.load_classifier(branch=self.branch.get())
        except FileNotFoundError as exc:
            messagebox.showinfo("No formulas", str(exc))
            return
        self.on_event({'type': 'distilled', 'meta': payload.get('meta', {}),
                       'result': None, 'formulas': payload['formulas']})

    def on_event(self, event):
        if event['type'] == 'progress':
            self._on_progress(event)
        elif event['type'] == 'distilled':
            meta = event['meta']
            metrics = meta.get('metrics', {}).get('test', {})
            oracle_bal = metrics.get('oracle_bal_acc', 0) or 1
            saved = event.get('saved') or {}
            note = f"\n\nin the library:\n{meta.get('library_name', '—')}"
            if meta.get('stopped'):
                note += "\n\n(interrupted manually —\ndid not become active)"
            elif saved and not saved.get('promoted'):
                note += f"\n\nactive not touched:\n{saved.get('reason', '')}"
            self.info.config(
                text=f"nodes: {meta.get('total_nodes', 0)}\n"
                     f"depth: {meta.get('max_depth', 0)}\n"
                     f"inputs: {meta.get('n_inputs_used', 0)}"
                     f"/{meta.get('n_inputs_available', 0)}\n\n"
                     f"test EML:    {metrics.get('eml_bal_acc', 0):.4f}\n"
                     f"test oracle: {metrics.get('oracle_bal_acc', 0):.4f}\n"
                     f"share: {metrics.get('eml_bal_acc', 0) / oracle_bal * 100:.1f}%\n"
                     f"fidelity: {metrics.get('fidelity', 0):.4f}{note}")
            self.text.delete('1.0', tk.END)
            if meta.get('class_layout'):
                # Branch C: there is one formula, and the main result is where
                # it placed the classes on its own axis.
                self.text.insert(tk.END, f"f =\n  {event['formulas'][0]}\n\n")
                self.text.insert(
                    tk.END, f"axis intervals: {meta.get('n_intervals', '?')}\n"
                            f"{meta['class_layout']}\n")
            else:
                for c, formula in enumerate(event['formulas']):
                    name = (config.CLASS_NAMES[c] if c < config.N_CLASSES
                            else f'C{c}')
                    self.text.insert(tk.END, f"f_{name} =\n  {formula}\n\n")
            if event.get('result'):
                self._draw_result(event['result'])
        elif event['type'] == 'error':
            messagebox.showerror("Distillation error", event['text'])
            self.info.config(text="error")
        elif event['type'] == 'done':
            self.btn.config(state=tk.NORMAL, text="Distill")
            self.btn_stop.config(state=tk.DISABLED)

    def _on_progress(self, event):
        """Live curves: the DATA phase accumulates one per class, ACC one."""
        if event['phase'] == 'single':
            self.single_hist = event['history']
            self.island_hist = event.get('islands') or []
            spread = (max(self.island_hist) - min(self.island_hist)
                      if len(self.island_hist) > 1 else 0.0)
            self.info.config(
                text=f"branch C\n"
                     f"generation {event['gen']}/{event['generations']}\n"
                     f"fitness: {event['best']:.4f}\n"
                     f"nodes: {event['nodes']}, depth {event['depth']}\n"
                     f"intervals: {event['intervals']}\n"
                     f"islands: {len(self.island_hist)}\n"
                     f"spread: {spread:.4f}")
        elif event['phase'] == 'data':
            self.data_hist[event['class_idx']] = event['history']
            variant = (f"variant {event['variant'] + 1}/{event['variants']}  "
                       if 'variant' in event else "")
            self.info.config(
                text=f"{variant}DATA phase\n"
                     f"{event['class_name']} "
                     f"({event['class_idx'] + 1}/{event['n_classes']})\n"
                     f"generation {event['gen'] + 1}/{event['generations']}\n"
                     f"fitness: {event['best']:.3f}\n"
                     f"nodes: {event['nodes']}, depth {event['depth']}")
        else:
            self.acc_hist = event['history']
            self.info.config(
                text=f"ACC phase\n"
                     f"generation {event['gen'] + 1}/{event['generations']}\n"
                     f"bal.accuracy: {event['best']:.4f}\n"
                     f"nodes: {event['nodes']}")

        now = time.perf_counter()
        if now - self._last_draw > self.REDRAW_INTERVAL:
            self._last_draw = now
            self._draw(self.data_hist, self.acc_hist)

    def _draw_result(self, result):
        """Final curves from a finished run (or from "Show saved")."""
        if 'model' in result:                     # branch C: a single curve
            self.single_hist = result.get('history') or []
            self._draw({}, [])
            return
        data_hist = {c: res['history']
                     for c, res in enumerate(result.get('data_results') or [])}
        acc = result.get('acc_result')
        self._draw(data_hist, acc['history'] if acc else [])

    def _draw(self, data_hist: dict, acc_hist: list):
        if self.single_hist:
            self.plot.clear()
            ax = self.plot.axes(111)
            ax.plot(self.single_hist, color=GREEN)
            ax.set_title("branch C: fitness (bal. accuracy − penalties)")
            ax.set_xlabel("island exchange round")
            self.plot.fig.tight_layout()
            self.plot.draw()
            return
        if not data_hist and not acc_hist:
            return
        self.plot.clear()
        has_acc = bool(acc_hist)
        ax = self.plot.axes(1, 2, 1) if has_acc else self.plot.axes(111)
        for c in sorted(data_hist):
            name = config.CLASS_NAMES[c] if c < config.N_CLASSES else f'C{c}'
            ax.plot(data_hist[c],
                    color=np.array(vision.class_color(c)) / 255, label=name)
        ax.set_title("DATA phase: regression fitness")
        ax.set_xlabel("generation")
        if data_hist:
            leg = ax.legend(fontsize=7, facecolor=BG2, edgecolor=BG3)
            for text in leg.get_texts():
                text.set_color(FG2)

        if has_acc:
            ax2 = self.plot.axes(1, 2, 2)
            ax2.plot(acc_hist, color=GREEN)
            ax2.set_title("ACC phase: balanced accuracy")
            ax2.set_xlabel("generation")
        self.plot.fig.tight_layout()
        self.plot.draw()


# ── Tab: Models ───────────────────────────────────────────────────────────────

ACTIVE_CHOICE = "— active —"


def _lib_branch(kind: str, branch: str) -> str:
    """
    Under which branch to look for a model of this kind in the library.

    Branch C has no oracle of its own: it learns from the same MLP as branch A
    and in its feature space. A request for "the branch C oracle" must therefore
    be redirected to branch A, otherwise the model lists would be empty and the
    "formula against network" comparison unavailable exactly where it is needed
    most.
    """
    return 'A' if kind == 'oracle' and branch.upper() == 'C' else branch


def _model_choices(kind: str, branch: str) -> list[str]:
    """
    List for the dropdown: "active" plus the library models.

    The first item is always the active model — in most cases that is the one
    wanted, and a specific version is chosen when comparing two.
    """
    import storage
    return [ACTIVE_CHOICE] + [e.label() for e in storage.list_models(kind, branch)]


def _resolve_choice(choice: str, kind: str, branch: str) -> str | None:
    """Menu item -> path to the model file (None = the active slot)."""
    import storage
    if not choice or choice == ACTIVE_CHOICE:
        return None
    for entry in storage.list_models(kind, branch):
        if entry.label() == choice:
            return entry.path
    return None                      # the model was deleted — silently take active


class ModelsTab:
    """
    Model library: what is saved, what is active, what is best.

    Training and distillation cost from seconds to an hour, and the GA spread
    across seeds exceeds the difference between method changes — so "which model
    are we measuring right now" is not an interface detail. All saved models are
    visible here with their characteristics, and any of them can be made active.
    """

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        self.root = root
        self.entries: list = []

        panel = tk.Frame(parent, bg=BG2, width=250)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "LIBRARY")
        _btn(panel, "Refresh list", self.refresh, fill=tk.X, padx=6, pady=3)
        self.btn_activate = _btn(panel, "Make active", self.activate,
                                 bg=GREEN, fg=BG, fill=tk.X, padx=6, pady=3)
        self.btn_rename = _btn(panel, "Rename...", self.rename,
                               fill=tk.X, padx=6, pady=3)
        self.btn_delete = _btn(panel, "Delete...", self.delete, bg=RED, fg=BG,
                               fill=tk.X, padx=6, pady=3)
        _sep(panel)
        _label(panel, "★ best by val\n• currently active", fg=FG2,
               fill=tk.X, padx=8, pady=(0, 4))
        _sep(panel)
        self.info = _label(panel, "select a model in the list", fill=tk.X,
                           padx=8, pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _label(right, "saved models", fg=ACCENT, font=MONO_B,
               fill=tk.X, padx=8, pady=(8, 2))

        columns = ('marks', 'kind', 'branch', 'name', 'val', 'test', 'size',
                   'when')
        self.tree = ttk.Treeview(right, columns=columns, show='headings',
                                 selectmode='browse', height=18)
        for column, title, width, anchor in (
                ('marks', '', 40, 'center'),
                ('kind', 'kind', 80, 'w'),
                ('branch', 'branch', 55, 'center'),
                ('name', 'name', 300, 'w'),
                ('val', 'val bal', 80, 'e'),
                ('test', 'test bal', 80, 'e'),
                ('size', 'complexity', 110, 'e'),
                ('when', 'saved', 120, 'w')):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, anchor=anchor)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.tree.bind('<<TreeviewSelect>>', lambda e: self._on_select())
        self.tree.tag_configure('best', foreground=GREEN)
        self.tree.tag_configure('active', foreground=ACCENT)

        _label(right, "The active model is the one used by the C export, "
                      "verify-c, bench and the test\ntabs by default. Runs enter "
                      "the library on their own; the best is chosen by val.",
               fg=FG2, fill=tk.X, padx=10, pady=(0, 8))

        if lock is not None:
            lock.subscribe(self._on_lock)
        self.refresh()

    def _on_lock(self, owner):
        """While training/distillation runs, model files are left alone."""
        state = tk.DISABLED if owner is not None else tk.NORMAL
        for button in (self.btn_activate, self.btn_rename, self.btn_delete):
            button.config(state=state)

    def refresh(self):
        import storage

        self.tree.delete(*self.tree.get_children())
        self.entries = storage.list_models()
        for i, entry in enumerate(self.entries):
            marks = ('★' if entry.is_best else '') + ('•' if entry.is_active else '')
            unit = 'par.' if entry.kind == 'oracle' else 'nodes'
            tags = []
            if entry.is_active:
                tags.append('active')
            elif entry.is_best:
                tags.append('best')
            self.tree.insert(
                '', tk.END, iid=str(i), tags=tags, values=(
                    marks,
                    'oracle' if entry.kind == 'oracle' else 'formulas',
                    entry.branch,
                    entry.name,
                    f"{entry.val_score:.4f}" if entry.val_score is not None else '—',
                    f"{entry.test_score:.4f}" if entry.test_score is not None else '—',
                    f"{entry.size} {unit}" if entry.size is not None else '—',
                    time.strftime('%d.%m %H:%M', time.localtime(entry.saved_at))))
        if not self.entries:
            self.info.config(text="the library is empty.\n\nModels get here\n"
                                  "on their own during training\nand distillation.")

    def _selected(self):
        selection = self.tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        return self.entries[index] if index < len(self.entries) else None

    def _on_select(self):
        entry = self._selected()
        if entry is not None:
            self.info.config(text=entry.details())

    def activate(self):
        import storage

        entry = self._selected()
        if entry is None:
            messagebox.showinfo("Nothing selected", "Select a model in the list.")
            return
        if entry.is_active:
            messagebox.showinfo("Already active",
                                f"\"{entry.name}\" is already active.")
            return
        storage.activate(entry)
        self.refresh()
        self.info.config(text=f"now active:\n\"{entry.name}\"")

    def rename(self):
        import storage
        from tkinter import simpledialog

        entry = self._selected()
        if entry is None:
            messagebox.showinfo("Nothing selected", "Select a model in the list.")
            return
        new_name = simpledialog.askstring("Rename",
                                          "New model name:",
                                          initialvalue=entry.name,
                                          parent=self.root)
        if not new_name or new_name == entry.name:
            return
        renamed = storage.rename_model(entry, new_name)
        self.refresh()
        self.info.config(text=f"new name:\n\"{renamed.name}\"")

    def delete(self):
        import storage

        entry = self._selected()
        if entry is None:
            messagebox.showinfo("Nothing selected", "Select a model in the list.")
            return
        note = ("\n\nThis model is currently active. The active slot will stay "
                "in place, only the library copy will disappear."
                if entry.is_active else "")
        if not messagebox.askyesno("Delete model",
                                   f"Delete \"{entry.name}\" permanently?{note}"):
            return
        storage.delete_model(entry)
        self.refresh()
        self.info.config(text="deleted")


# ── Tab: Test · Photo ─────────────────────────────────────────────────────────

class PhotoTab:
    """
    Classification of a single image: the formula answer next to the oracle's.

    The point of the tab is not to "show a prediction" but to compare two
    engines on the same frame: where they agree, where they diverge, and how
    confident they are.
    """

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        self.root = root
        self.images = self.labels = None
        self.crop = None
        self.true_label = None
        self._predictors: dict[tuple, vision.Predictor] = {}

        panel = tk.Frame(parent, bg=BG2, width=215)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "BRANCH")
        self.branch = tk.StringVar(value='A')
        for value, text in (('A', "branch A"), ('B', "branch B"),
                            ('C', "branch C (one formula)")):
            tk.Radiobutton(panel, text=text, variable=self.branch,
                           value=value, bg=BG2, fg=FG, selectcolor=BG3,
                           font=("Consolas", 8), activebackground=BG2,
                           anchor='w',
                           command=self._on_branch).pack(fill=tk.X, padx=6)

        # Selection of specific models. Comparing "formulas against the oracle"
        # and "this version of the formulas against that one" is the same
        # operation, so the engines are chosen independently.
        _section(panel, "MODELS")
        _label(panel, "oracle:", fill=tk.X, padx=8)
        self.pick_oracle = tk.StringVar(value=ACTIVE_CHOICE)
        self.combo_oracle = ttk.Combobox(panel, textvariable=self.pick_oracle,
                                         font=("Consolas", 7), state='readonly')
        self.combo_oracle.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.combo_oracle.bind('<<ComboboxSelected>>', lambda e: self.classify())

        _label(panel, "formulas:", fill=tk.X, padx=8)
        self.pick_eml = tk.StringVar(value=ACTIVE_CHOICE)
        self.combo_eml = ttk.Combobox(panel, textvariable=self.pick_eml,
                                      font=("Consolas", 7), state='readonly')
        self.combo_eml.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.combo_eml.bind('<<ComboboxSelected>>', lambda e: self.classify())
        _btn(panel, "Refresh model list", self.refresh_models,
             fill=tk.X, padx=6, pady=3)

        _section(panel, "SOURCE")
        self.btn_random = _btn(panel, "Random from dataset", self.random_sample,
                               fill=tk.X, padx=6, pady=3)
        _label(panel, "class:", fill=tk.X, padx=8, pady=(4, 0))
        self.class_filter = tk.StringVar(value="any")
        ttk.Combobox(panel, textvariable=self.class_filter, font=("Consolas", 8),
                     values=["any"] + config.CLASS_NAMES, state='readonly'
                     ).pack(fill=tk.X, padx=6, pady=3)
        self.btn_open = _btn(panel, "Open file...", self.open_file,
                             fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "no image selected", fill=tk.X, padx=8,
                           pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top = tk.Frame(right, bg=BG)
        top.pack(fill=tk.X, padx=8, pady=8)
        self.view = ImageView(top, width=256, height=256)
        self.view.pack(side=tk.LEFT)
        self.verdict = _label(top, "", fg=FG, font=("Consolas", 11, "bold"),
                              side=tk.LEFT, padx=16, anchor='n')

        self.plot = PlotPanel(right, "gui_photo_scores", figsize=(7.5, 3.2))
        self.plot.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        if lock is not None:
            lock.subscribe(self._on_lock)
        self.refresh_models()

    def _on_lock(self, owner):
        """While training/distillation runs, models are not read from disk."""
        state = tk.DISABLED if owner is not None else tk.NORMAL
        self.btn_random.config(state=state)
        self.btn_open.config(state=state)
        if owner is None:
            # The run finished — a new model appeared in the library.
            self.refresh_models()

    def refresh_models(self):
        """Re-read the library into both dropdowns."""
        branch = self.branch.get()
        for combo, var, kind in ((self.combo_oracle, self.pick_oracle, 'oracle'),
                                 (self.combo_eml, self.pick_eml, 'eml')):
            choices = _model_choices(kind, _lib_branch(kind, branch))
            combo['values'] = choices
            if var.get() not in choices:
                var.set(ACTIVE_CHOICE)

    def _on_branch(self):
        """Branch switch: it has its own models, the list must be rebuilt."""
        self.refresh_models()
        self.classify()

    def _predictor(self, engine: str) -> vision.Predictor:
        """
        Models are cached: reloading them on every click is pointless.

        But the cache must notice retraining. Previously the key was the pair
        (branch, engine), and after training or distillation in a neighboring
        tab the photo test kept silently showing the answers of the OLD model —
        i.e. comparing something other than what is on disk. The key therefore
        also includes the file modification time: the checkpoint was rewritten —
        the key differs — the model is loaded anew.
        """
        import storage

        branch = self.branch.get()
        kind = 'oracle' if engine == 'oracle' else 'eml'
        lib_branch = _lib_branch(kind, branch)
        choice = (self.pick_oracle if engine == 'oracle' else self.pick_eml).get()
        chosen = _resolve_choice(choice, kind, lib_branch)

        # Branch B with formulas depends on TWO files: the formula input is the
        # CNN embedding, so retraining the B oracle changes the EML engine's
        # answer exactly as a new distillation does.
        paths = [chosen or storage.active_path(kind, lib_branch)]
        if engine == 'eml' and branch == 'B':
            paths.append(storage.oracle_path('B'))

        stamps = []
        for path in paths:
            try:
                stamps.append(os.path.getmtime(path))
            except OSError:
                stamps.append(None)
        key = (branch, engine, chosen, tuple(stamps))
        if key not in self._predictors:
            predictor = vision.Predictor(branch=branch, engine=engine,
                                         path=chosen)
            # There is no reason to keep every version in a row — only the
            # freshest is of interest.
            self._predictors = {k: v for k, v in self._predictors.items()
                                if k[:2] != (branch, engine)}
            self._predictors[key] = predictor
        return self._predictors[key]

    def random_sample(self):
        import data
        if self.images is None:
            self.images, self.labels, _ = data.load_dataset(verbose=False)
        pool = np.arange(len(self.labels))
        if self.class_filter.get() != "any":
            c = config.CLASS_NAMES.index(self.class_filter.get())
            pool = np.flatnonzero(self.labels == c)
        i = int(np.random.default_rng().choice(pool))
        self.crop = self.images[i]
        self.true_label = int(self.labels[i])
        self.classify()

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.bmp *.png *.jpg *.jpeg"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            frame = _read_image(path)
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("Could not open", str(exc))
            return
        # The file may be of any size while the model was trained on 64x64 —
        # take the central window rather than stretching the whole frame:
        # stretching would distort the texture scale the features rely on.
        self.crop = vision.center_crop(frame)
        self.true_label = None
        self.classify()

    def classify(self):
        if self.crop is None:
            return
        self.view.show(vision.resize_nearest(self.crop, 256, 256))

        results = {}
        for engine in ('oracle', 'eml'):
            try:
                predictor = self._predictor(engine)
            except FileNotFoundError:
                continue
            except Exception as exc:               # noqa: BLE001
                # A corrupted formula JSON, a dimension mismatch after a feature
                # set change, a damaged checkpoint — all of these used to fly as
                # an unhandled exception into the main thread and kill the
                # button click silently, without a word in the window.
                messagebox.showerror(
                    "Could not load the model",
                    f"{'formulas' if engine == 'eml' else 'oracle'}, "
                    f"branch {self.branch.get()}:\n{exc}")
                continue
            preds, scores, conf = predictor.predict(self.crop[None])
            results[engine] = (int(preds[0]), scores[0], float(conf[0]),
                               predictor.label(), predictor.conf_is_probability,
                               predictor)
        if not results:
            messagebox.showinfo("No models",
                                f"There is no trained model or formulas for "
                                f"branch {self.branch.get()}.")
            return

        lines = []
        if self.true_label is not None:
            lines.append(f"truth:   {config.CLASS_NAMES[self.true_label]}")
        for engine, (pred, _, conf, label, is_prob, predictor) in results.items():
            mark = ''
            if self.true_label is not None:
                mark = '  correct' if pred == self.true_label else '  WRONG'
            # For the oracle this is a probability after softmax, for the
            # formulas just a score: their outputs were never calibrated as
            # probabilities. Labeling two different numbers identically is not
            # allowed, and how exactly to label them is known by the engine
            # itself rather than by this tab.
            suffix = (f"{conf * 100:4.0f}%" if is_prob
                      else f"{predictor.conf_label()} {conf:.2f}")
            lines.append(f"{'formulas' if engine == 'eml' else 'oracle':<9} "
                         f"{config.CLASS_NAMES[pred]:<7} {suffix}{mark}")
        if len(results) == 2:
            agree = results['eml'][0] == results['oracle'][0]
            lines.append("")
            lines.append("engines agree" if agree else "engines DIVERGE")
        lines.append("")
        lines.append("oracle % is a probability,")
        eml_predictor = results.get('eml', (None,) * 6)[5]
        if eml_predictor is not None and eml_predictor.is_single:
            lines.append("margin — share of the axis interval")
            lines.append("to the nearest threshold, 0..1")
        elif eml_predictor is not None:
            lines.append("formula score is not a probability")
            lines.append(f"(minimum for {config.N_CLASSES} classes "
                         f"{eml_predictor.conf_floor:.2f})")
        self.verdict.config(text="\n".join(lines))
        self.info.config(text=f"frame {self.crop.shape[1]}x{self.crop.shape[0]}\n"
                              f"branch {self.branch.get()}")
        self._draw_scores(results)

    def _draw_scores(self, results):
        self.plot.clear()
        ax = self.plot.axes(111)
        width = 0.38
        x = np.arange(config.N_CLASSES)
        for k, item in enumerate(results.items()):
            engine, (_, scores, _, label, _is_prob, predictor) = item
            if predictor.is_single:
                # Branch C: a class score is minus the distance along the output
                # axis, and it has no upper bound. A softmax of such numbers
                # saturates to one almost always, so the bar is drawn
                # differently: 1 for the predicted class, and for the rest how
                # close they were, as a fraction of the most distant one.
                worst = float(np.min(scores))
                probs = (1.0 - scores / worst) if worst < 0 else np.ones_like(scores)
            else:
                # The engine scores live on different scales (logits against
                # calibrated formula outputs), so they can only be compared
                # after a softmax — otherwise the bars are incomparable.
                probs = vision.softmax(scores[None])[0]
            ax.bar(x + (k - 0.5) * width, probs, width,
                   label=label, color=ACCENT if engine == 'eml' else PURPLE)
        ax.set_xticks(x)
        ax.set_xticklabels(config.CLASS_NAMES, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("class scores after softmax "
                     "(not probabilities for the formulas)")
        leg = ax.legend(fontsize=7, facecolor=BG2, edgecolor=BG3)
        for text in leg.get_texts():
            text.set_color(FG2)
        self.plot.fig.tight_layout()
        self.plot.draw()


def _read_image(path: str) -> np.ndarray:
    """
    Read an image into RGB.

    BMP and PNG are read by the own parsers (imaging.py) and require nothing
    third-party; JPEG — only through OpenCV or Pillow, if they are installed.
    """
    import imaging
    return imaging.read_image(path)


# ── Tab: Test · Video ─────────────────────────────────────────────────────────

class VideoTab(WorkerTab):
    """
    Real-time video annotation.

    The frame is cut into 64x64 windows — exactly the size the model was trained
    on — each window is classified, and a semi-transparent fill in the class
    color is laid over the frame.

    Decoding and inference run in a worker thread, finished frames go into a
    queue, and the main thread draws them. Otherwise the interface would freeze
    for the duration of each frame's processing.
    """

    MAX_QUEUE = 4        # no point buffering more frames: the video is live

    # Recording buffer cap. A 900x560 RGB frame is ~1.5 MB, at 12 frames/s that
    # is almost 18 MB per second. Without a cap, ten minutes of recording is
    # gigabytes in a Python list and the application dies on memory for no
    # reason. 600 frames ~ a minute of video and under a gigabyte; beyond that
    # recording stops with a warning in the panel rather than silently eating
    # the machine.
    MAX_RECORD_FRAMES = 600
    MAX_RECORD_BYTES = 900 * 1024 * 1024

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        super().__init__(root)
        self.frames: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE)
        self.recording: list[np.ndarray] = []
        self.recording_bytes = 0
        self.record_full = False
        self.last_frame = None
        self._fps_times: list[float] = []
        self._running_model = None        # (branch, engine) of the current source
        self._source = None               # what to restart with on a model change
        # Snapshot of the overlay settings as plain numbers.
        #
        # Reading tkinter variables from a worker thread is NOT allowed: Tk is
        # not thread-safe and answers "main thread is not in main loop". It
        # fails intermittently rather than always, which makes it hard to catch.
        # The values are therefore snapshotted in the main thread
        # (_refresh_settings, called from _tick), and the video thread reads only
        # this dict.
        self._live = {'stride': vision.DEFAULT_STRIDE, 'alpha': 0.35,
                      'conf': 0.0}

        panel = tk.Frame(parent, bg=BG2, width=215)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "SOURCE")
        self.src_buttons = [
            _btn(panel, "Video file...", self.open_file, fill=tk.X, padx=6, pady=3),
            _btn(panel, "Webcam", self.open_camera, fill=tk.X, padx=6, pady=3),
            _btn(panel, "Synthetic from dataset", self.open_synthetic,
                 fill=tk.X, padx=6, pady=3),
        ]

        _section(panel, "MODEL")
        self.branch = tk.StringVar(value='A')
        self.engine = tk.StringVar(value='eml')
        # Switching the model during playback restarts the source. Previously
        # the radio buttons simply did nothing until the next start: the user
        # selected "oracle", saw the previous formula annotation, and honestly
        # believed it was the oracle's answer.
        self.model_radios = []
        for value, text in (('A', "branch A"), ('B', "branch B"),
                            ('C', "branch C")):
            self.model_radios.append(tk.Radiobutton(
                panel, text=text, variable=self.branch, value=value,
                bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                activebackground=BG2, anchor='w',
                command=self._on_model_change))
            self.model_radios[-1].pack(fill=tk.X, padx=6)
        for value, text in (('eml', "EML formulas"), ('oracle', "oracle")):
            self.model_radios.append(tk.Radiobutton(
                panel, text=text, variable=self.engine, value=value,
                bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                activebackground=BG2, anchor='w',
                command=self._on_model_change))
            self.model_radios[-1].pack(fill=tk.X, padx=6)

        self.pick_model = tk.StringVar(value=ACTIVE_CHOICE)
        self.combo_model = ttk.Combobox(panel, textvariable=self.pick_model,
                                        font=("Consolas", 7), state='readonly')
        self.combo_model.pack(fill=tk.X, padx=6, pady=4)
        self.combo_model.bind('<<ComboboxSelected>>',
                              lambda e: self._on_model_change())

        params = _section(panel, "OVERLAY")
        self.p_stride = ParamRow(params, "window step", vision.DEFAULT_STRIDE, int)
        self.p_alpha = ParamRow(params, "transparency", 0.35, float)
        self.p_conf = ParamRow(params, "conf. threshold", 0.0, float)
        self.show_labels = tk.BooleanVar(value=True)
        tk.Checkbutton(panel, text="class labels", variable=self.show_labels,
                       bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                       activebackground=BG2, anchor='w').pack(fill=tk.X, padx=6,
                                                              pady=3)

        _sep(panel)
        self.btn_stop = _btn(panel, "Stop", self.stop, bg=RED, fg=BG,
                             state=tk.DISABLED, fill=tk.X, padx=6, pady=3)
        _btn(panel, "Save frame", self.save_frame, fill=tk.X, padx=6, pady=3)
        self.rec_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="record result", variable=self.rec_var,
                       bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                       activebackground=BG2, anchor='w').pack(fill=tk.X, padx=6)
        _btn(panel, "Save video", self.save_video, fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "no source selected", fill=tk.X, padx=8, pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.view = ImageView(right, width=900, height=560)
        self.view.pack(padx=8, pady=8)
        _color_legend(right).pack(fill=tk.X, padx=10)

        if lock is not None:
            lock.subscribe(self._on_lock)
        self.refresh_models()
        self._tick()

    def _on_lock(self, owner):
        """While training/distillation runs, no new source is started."""
        state = tk.DISABLED if owner is not None else tk.NORMAL
        for button in self.src_buttons:
            button.config(state=state)
        if owner is None:
            self.refresh_models()

    def refresh_models(self):
        """Model list for the current (branch, engine) pair."""
        kind = 'oracle' if self.engine.get() == 'oracle' else 'eml'
        choices = _model_choices(kind, _lib_branch(kind, self.branch.get()))
        self.combo_model['values'] = choices
        if self.pick_model.get() not in choices:
            self.pick_model.set(ACTIVE_CHOICE)

    def _current_model(self) -> tuple[str, str, str | None]:
        """(branch, engine, path) — what should annotate the image right now."""
        branch, engine = self.branch.get(), self.engine.get()
        kind = 'oracle' if engine == 'oracle' else 'eml'
        return branch, engine, _resolve_choice(self.pick_model.get(), kind,
                                               _lib_branch(kind, branch))

    def _on_model_change(self):
        """Changing the model on the fly = restarting the source with it."""
        self.refresh_models()
        if self._running_model is None or not self.busy:
            return
        if self._running_model == self._current_model():
            return
        self._start_source(self._source)

    # ── Sources ───────────────────────────────────────────────────────────
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        if path:
            self._start_source(path)

    def open_camera(self):
        self._start_source(0)

    def open_synthetic(self):
        """
        Video assembled from dataset crops.

        Needed so that video mode can be checked without aerial footage at hand:
        every window here has a known ground truth label, so the corner shows the
        actual annotation accuracy rather than just pretty boxes.
        """
        self._start_source('synthetic')

    def _start_source(self, source):
        # The old thread must not only be asked to stop but also waited for:
        # otherwise start() sees it alive, decides it is busy, and the source
        # switch silently fails.
        self.request_stop()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        _drain(self.frames)
        self._refresh_settings()      # so the very first frame is computed right
        branch, engine, path = self._current_model()
        if not self.start(self._work, source, branch, engine, path):
            return
        self._source = source
        self._running_model = (branch, engine, path)
        self.btn_stop.config(state=tk.NORMAL)
        self.recording = []
        self.recording_bytes = 0
        self.record_full = False
        self._fps_times = []
        self.info.config(text=f"source: {source}\n"
                              f"model: {branch}/{engine}\nstarting...")

    def stop(self):
        self.request_stop()
        self._running_model = None
        self.btn_stop.config(state=tk.DISABLED)

    # ── Worker thread ─────────────────────────────────────────────────────
    def _work(self, source, branch, engine, path=None):
        predictor = vision.Predictor(branch=branch, engine=engine, path=path)
        video = None
        if source == 'synthetic':
            stream = self._synthetic_stream()
            fps_limit = 0.0
        else:
            video = vision.VideoSource(source)
            stream = self._video_stream(video)
            # A file plays at its own pace, a camera as fast as it manages.
            fps_limit = video.fps if not video.is_camera else 0.0

        try:
            for frame, truth in stream:
                if self.should_stop():
                    break
                t0 = time.perf_counter()
                settings = dict(self._live)      # snapshot made by the main thread
                positions = vision.tile_positions(frame.shape[0], frame.shape[1],
                                                  stride=settings['stride'])
                if not positions:
                    continue
                tiles = vision.extract_tiles(frame, positions)
                preds, _, conf = predictor.predict(tiles)
                overlaid = vision.draw_overlay(
                    frame, positions, preds, conf, alpha=settings['alpha'],
                    conf_threshold=settings['conf'])
                infer_ms = (time.perf_counter() - t0) * 1000

                accuracy = None
                if truth is not None:
                    accuracy = float((preds == truth).mean())

                try:
                    self.frames.put(
                        {'frame': overlaid, 'positions': positions,
                         'preds': preds, 'conf': conf, 'infer_ms': infer_ms,
                         'accuracy': accuracy, 'n_tiles': len(positions)},
                        timeout=0.5)
                except queue.Full:
                    # The interface is not keeping up — the frame is dropped.
                    # For live video that is better than accumulating lag.
                    pass

                if fps_limit > 0:
                    time.sleep(max(0.0, 1.0 / fps_limit - (time.perf_counter() - t0)))
        finally:
            if video is not None:
                video.release()

    def _video_stream(self, video):
        while True:
            frame = video.read()
            if frame is None:
                if video.is_camera:
                    break
                video.rewind()                        # a file loops
                frame = video.read()
                if frame is None:
                    break
            yield frame, None

    def _synthetic_stream(self):
        """A slowly scrolling mosaic of crops with known labels."""
        import data

        images, labels, _ = data.load_dataset(verbose=False)
        _, _, test_idx = data.stratified_split(labels)
        rng = np.random.default_rng(0)
        rows, cols = 7, 12
        pool = test_idx.copy()
        rng.shuffle(pool)

        cursor = 0
        while True:
            if self.should_stop():
                return
            take = rows * cols
            if cursor + take > len(pool):
                rng.shuffle(pool)
                cursor = 0
            sel = pool[cursor:cursor + take]
            cursor += cols                            # shift by a row per frame
            frame = vision.mosaic(images[sel], rows, cols)
            yield frame, labels[sel]
            time.sleep(1 / 12)

    # ── Main thread: rendering ────────────────────────────────────────────
    def _refresh_settings(self):
        """
        Move the field values into a plain dict.

        Called only from the main thread — this is the only place where the
        tkinter overlay variables are read.
        """
        self._live = {
            'stride': max(8, self.p_stride.get(vision.DEFAULT_STRIDE)
                          or vision.DEFAULT_STRIDE),
            'alpha': float(np.clip(self.p_alpha.get(0.35) or 0.0, 0.0, 1.0)),
            'conf': float(self.p_conf.get(0.0) or 0.0),
        }

    def _tick(self):
        self._refresh_settings()
        try:
            payload = self.frames.get_nowait()
        except queue.Empty:
            payload = None

        if payload is not None:
            self._render(payload)
        self.root.after(15, self._tick)

    def _render(self, payload):
        frame = payload['frame']
        self.last_frame = frame
        if self.rec_var.get():
            self._record(frame)

        labels = []
        if self.show_labels.get():
            for i, (y, x) in enumerate(payload['positions']):
                cls = int(payload['preds'][i])
                labels.append((x, y, config.CLASS_NAMES[cls][:4],
                               '#%02x%02x%02x' % vision.class_color(cls)))
        self.view.show(frame, labels)

        now = time.perf_counter()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 2.0]
        # Dividing by the window length (2 s) will not do: until the window
        # fills up the counter understates — in the first second it showed half
        # the real FPS. We use the time actually elapsed: N-1 intervals between
        # N frames.
        span = self._fps_times[-1] - self._fps_times[0]
        fps = (len(self._fps_times) - 1) / span if span > 1e-9 else 0.0

        text = (f"windows: {payload['n_tiles']}\n"
                f"inference: {payload['infer_ms']:.0f} ms\n"
                f"FPS: {fps:.1f}\n"
                f"frame: {frame.shape[1]}x{frame.shape[0]}")
        if self._running_model:
            branch, engine, _ = self._running_model
            text += f"\nmodel: {branch}/{engine}"
            if self.pick_model.get() != ACTIVE_CHOICE:
                text += f"\n{self.pick_model.get().strip()[:22]}"
        if payload['accuracy'] is not None:
            text += f"\n\nannotation accuracy:\n{payload['accuracy'] * 100:.1f}%"
        if self.rec_var.get() or self.recording:
            text += (f"\n\nrecorded: {len(self.recording)} frames"
                     f"\n({self.recording_bytes / 1024 / 1024:.0f} MB)")
            if self.record_full:
                text += "\nBUFFER FULL — recording\nstopped"
        self.info.config(text=text)

    def _record(self, frame: np.ndarray):
        """Append a frame to the recording buffer if it is not full yet."""
        if self.record_full:
            return
        if (len(self.recording) >= self.MAX_RECORD_FRAMES
                or self.recording_bytes + frame.nbytes > self.MAX_RECORD_BYTES):
            self.record_full = True
            messagebox.showwarning(
                "Recording buffer full",
                f"Recorded {len(self.recording)} frames "
                f"({self.recording_bytes / 1024 / 1024:.0f} MB) — the limit.\n\n"
                f"Frames accumulate in memory, so the buffer has a cap "
                f"({self.MAX_RECORD_FRAMES} frames or "
                f"{self.MAX_RECORD_BYTES // 1024 // 1024} MB).\n"
                f"Save the video and clear \"record result\" to start a new "
                f"recording.")
            return
        self.recording.append(frame)
        self.recording_bytes += frame.nbytes

    # ── Saving ────────────────────────────────────────────────────────────
    def save_frame(self):
        """
        Save the annotated frame AS IS, pixel for pixel.

        Previously the frame went through a matplotlib figure with dpi=100 and
        was saved with dpi=150: an image one and a half times larger than the
        original and resampled landed on disk — i.e. not the pixels the user saw
        and judged. For a report where annotation is counted from that frame,
        that is unacceptable.
        """
        if self.last_frame is None:
            messagebox.showinfo("No frame", "Start a source first.")
            return
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR,
                            f"overlay_{time.strftime('%Y%m%d_%H%M%S')}.png")
        try:
            written = vision.write_image(path, self.last_frame)
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("Could not save the frame", str(exc))
            return
        h, w = self.last_frame.shape[:2]
        self.info.config(text=f"frame {w}x{h} -> {written}")

    def save_video(self):
        if not self.recording:
            messagebox.showinfo(
                "Nothing to save",
                "Enable \"record result\" and start a source.")
            return
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR,
                            f"overlay_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        try:
            vision.write_video(path, self.recording)
        except vision.VideoUnavailable as exc:
            messagebox.showerror("No OpenCV", str(exc))
            return
        self.info.config(text=f"video ({len(self.recording)} frames)\n-> {path}")

    def on_event(self, event):
        if event['type'] == 'error':
            messagebox.showerror("Video error", event['text'])
            self.info.config(text="error")
            self.btn_stop.config(state=tk.DISABLED)
        elif event['type'] == 'done':
            self.btn_stop.config(state=tk.DISABLED)


# ── Tab: Test (container) ─────────────────────────────────────────────────────

class TestTab:
    """Container with the two test modes — photo and video."""

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)
        photo = tk.Frame(nb, bg=BG)
        video = tk.Frame(nb, bg=BG)
        nb.add(photo, text="  Photo  ")
        nb.add(video, text="  Video  ")
        self.photo = PhotoTab(root, photo, lock=lock)
        self.video = VideoTab(root, video, lock=lock)


# ── Notebook styling ──────────────────────────────────────────────────────────

def _style_notebook(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("default")
    except tk.TclError:
        pass
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=BG2, foreground=FG2,
                    font=("Consolas", 10, "bold"), padding=(18, 7), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", BG3)],
              foreground=[("selected", ACCENT)])
    style.configure("TCombobox", fieldbackground=BG3, background=BG3,
                    foreground=FG, arrowcolor=FG2)
    # Model library table: the same dark background as the rest of the window.
    style.configure("Treeview", background=BG2, fieldbackground=BG2,
                    foreground=FG, borderwidth=0, font=("Consolas", 8),
                    rowheight=20)
    style.configure("Treeview.Heading", background=BG3, foreground=ACCENT,
                    font=("Consolas", 8, "bold"), borderwidth=0)
    style.map("Treeview", background=[("selected", BG3)],
              foreground=[("selected", ACCENT)])


# ── Entry point ───────────────────────────────────────────────────────────────

def main(initial_tab: str | None = None) -> None:
    # The dataset is applied BEFORE the tabs are created: the class names, their
    # count and all the captions the tabs build in their constructors depend on
    # it.
    config.load_dataset_choice()

    root = tk.Tk()
    root.title(f"EML Classifier — {config.dataset_summary()}")
    root.geometry("1280x820")
    root.configure(bg=BG)
    root.minsize(1040, 680)

    _style_notebook(root)
    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True)

    frames = {}
    for key, text in [("data", "  Data  "),
                      ("features", "  Features  "),
                      ("train", "  Training  "),
                      ("distill", "  Distillation  "),
                      ("models", "  Models  "),
                      ("test", "  Test  ")]:
        frames[key] = tk.Frame(nb, bg=BG)
        nb.add(frames[key], text=text)

    # One lock for the whole application: while one tab is busy with a heavy
    # task, the others do not allow starting theirs.
    lock = BusyLock()
    DataTab(root, frames["data"])
    FeaturesTab(root, frames["features"])
    TrainTab(root, frames["train"], lock=lock)
    DistillTab(root, frames["distill"], lock=lock)
    ModelsTab(root, frames["models"], lock=lock)
    TestTab(root, frames["test"], lock=lock)

    if initial_tab in frames:
        nb.select(frames[initial_tab])

    root.mainloop()


if __name__ == "__main__":
    main()

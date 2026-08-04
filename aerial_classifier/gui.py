"""
gui.py — интерфейс проекта: данные · признаки · обучение · дистилляция · тест.

Стиль и хелперы взяты из tetris/gui.py, чтобы подпроекты выглядели одинаково.

Вкладки:
    Данные        — состав датасета, примеры вырезок каждого класса.
    Признаки      — F-статистика и распределения признаков по классам.
    Обучение      — запуск обучения оракула, живые кривые loss / val accuracy.
    Дистилляция   — запуск GA, живые кривые обеих фаз, текст формул.
    Тест          — две вкладки внутри: Фото и Видео.

Каждый график можно сохранить кнопкой «Сохранить график» — файл ложится в
results/ с говорящим именем.

Три решения, которые стоит пояснить:

1. Тяжёлые операции (обучение, дистилляция, декодирование видео) выполняются
   в отдельных потоках, а результаты передаются в интерфейс через queue.Queue
   и разбираются в главном потоке по root.after(). tkinter не потокобезопасен:
   рисовать из рабочего потока — это редкие, невоспроизводимые падения.

   Прогресс идёт по тому же каналу: обучение и дистилляция принимают
   progress_cb и should_stop, поэтому кривые рисуются по ходу дела, а «Стоп»
   останавливает работу на ближайшей эпохе/поколении, сохраняя лучшее
   найденное. Раньше обе задачи молчали до самого конца — для дистилляции это
   были десятки минут неподвижного окна.

2. Кадры показываются через tk.Canvas и PhotoImage из формата PPM, собранного
   вручную из numpy. Так показ картинок не требует ни Pillow, ни OpenCV —
   а значит фото-режим работает даже там, где вкладка видео отключена.

3. Тяжёлая задача в приложении может идти только одна (BusyLock). Обучение
   переписывает models/oracle_X.pt, дистилляция его читает — запускать их
   одновременно значит читать чекпоинт в момент записи.
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


# ── Палитра Catppuccin Mocha (как в tetris/gui.py) ───────────────────────────
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


# ── Хелперы виджетов ──────────────────────────────────────────────────────────

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
    """Озаглавленный блок панели управления."""
    tk.Label(parent, text=title, bg=parent['bg'], fg=ACCENT,
             font=("Consolas", 9, "bold"), anchor='w').pack(fill=tk.X,
                                                            padx=6, pady=(8, 2))
    frame = tk.Frame(parent, bg=parent['bg'])
    frame.pack(fill=tk.X)
    return frame


class ParamRow:
    """Строка «Label + Entry» для одного параметра."""

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


# ── Показ картинок через PPM ──────────────────────────────────────────────────

def _to_photo(rgb: np.ndarray) -> tk.PhotoImage:
    """
    numpy (H, W, 3) uint8 -> tk.PhotoImage через формат PPM.

    Tk умеет читать PPM «из коробки», поэтому Pillow не нужен: собираем
    заголовок и сырые байты руками.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    header = f"P6\n{w} {h}\n255\n".encode('ascii')
    return tk.PhotoImage(data=header + rgb.tobytes())


def _drain(q: queue.Queue) -> None:
    """Опустошить очередь — чтобы при смене источника не показать чужие кадры."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _color_legend(parent: tk.Widget) -> tk.Frame:
    """Полоска «цвет — класс»: без неё оверлей читается как набор пятен."""
    row = tk.Frame(parent, bg=parent['bg'])
    tk.Label(row, text="классы:", bg=parent['bg'], fg=FG2,
             font=("Consolas", 8)).pack(side=tk.LEFT, padx=(0, 6))
    for i, name in enumerate(config.CLASS_NAMES):
        color = '#%02x%02x%02x' % vision.class_color(i)
        tk.Label(row, text="  ", bg=color).pack(side=tk.LEFT)
        tk.Label(row, text=f" {name}  ", bg=parent['bg'], fg=FG,
                 font=("Consolas", 8)).pack(side=tk.LEFT)
    return row


class ImageView(tk.Canvas):
    """
    Канва для показа кадра с подписями поверх него.

    Подписи рисуются элементами канвы, а не вжигаются в пиксели: так текст
    остаётся резким при любом масштабе и не портит сохраняемый кадр.
    """

    def __init__(self, parent, width=640, height=480):
        super().__init__(parent, bg=BG, highlightthickness=0,
                         width=width, height=height)
        self._photo = None          # ссылку надо держать, иначе Tk соберёт её

    def show(self, rgb: np.ndarray, labels=None):
        """
        Показать кадр. labels — список (x, y, текст, цвет) в пикселях кадра.
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


# ── Панель с графиком и кнопкой сохранения ────────────────────────────────────

class PlotPanel(tk.Frame):
    """
    matplotlib-фигура внутри окна плюс кнопка «Сохранить график».

    Тёмная тема задаётся здесь один раз, чтобы каждый график не настраивал
    цвета заново и все вкладки выглядели одинаково.
    """

    def __init__(self, parent, name: str, figsize=(7.5, 4.5)):
        super().__init__(parent, bg=BG)
        self.name = name
        self.fig = Figure(figsize=figsize, dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill=tk.X, pady=(2, 0))
        _btn(bar, "Сохранить график", self.save, side=tk.RIGHT, padx=4)
        self._status = _label(bar, "", fg=FG2, side=tk.RIGHT, padx=6)

    def axes(self, *args, **kwargs):
        """Создать оси, уже оформленные под тёмную тему."""
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


# ── Общая занятость приложения ────────────────────────────────────────────────

class BusyLock:
    """
    Одна тяжёлая задача на всё приложение.

    Обучение переписывает models/oracle_X.pt, дистилляция читает его же и
    пишет models/best_eml_X.json, обе трогают общие кэши. Ничто не мешало
    запустить их одновременно — и получить дистилляцию оракула, которого в
    этот момент перезаписывают. Поэтому вкладка, начинающая работу, забирает
    блокировку, а остальные на это время гасят кнопки запуска.

    Слушатели вызываются ТОЛЬКО из главного потока (acquire/release зовутся из
    обработчиков кнопок и из разбора очереди), иначе они трогали бы виджеты из
    чужого потока.
    """

    def __init__(self):
        self._owner: str | None = None
        self._listeners: list = []

    @property
    def owner(self) -> str | None:
        return self._owner

    def subscribe(self, callback):
        """callback(owner | None) — вызывается при смене владельца и сразу."""
        self._listeners.append(callback)
        callback(self._owner)

    def acquire(self, name: str) -> bool:
        if self._owner is not None:
            messagebox.showinfo(
                "Занято", f"Сейчас выполняется: {self._owner}.\n"
                          f"Дождитесь окончания или нажмите «Стоп».")
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


# ── Общий предок вкладок с фоновой задачей ────────────────────────────────────

class WorkerTab:
    """
    Вкладка, запускающая долгую операцию в отдельном потоке.

    Поток кладёт события в очередь, главный поток разбирает её по таймеру —
    рисовать из рабочего потока нельзя, tkinter этого не переживает.
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
            messagebox.showinfo("Занято", "Операция уже выполняется.")
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
            # В окне показывается только короткое сообщение, а полный стек
            # уходит в консоль: без него отладка фоновой задачи превращается
            # в гадание по одной строке.
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
                # Блокировка снимается здесь, а не в рабочем потоке: её
                # слушатели трогают виджеты, а это можно только отсюда.
                if event.get('type') == 'done' and self.lock is not None:
                    self.lock.release(self.lock_name)
                self.on_event(event)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def on_event(self, event: dict):
        """Переопределяется наследниками."""


# ── Вкладка: Данные ───────────────────────────────────────────────────────────

class DataTab:
    """Состав датасета и примеры вырезок каждого класса."""

    def __init__(self, root: tk.Tk, parent: tk.Frame):
        self.root = root
        self.images = self.labels = None

        panel = tk.Frame(parent, bg=BG2, width=210)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "ПАПКА С КЛАССАМИ")
        _btn(panel, "Выбрать папку...", self.choose_folder, bg=PURPLE, fg=BG,
             fill=tk.X, padx=6, pady=3)
        self.path_info = _label(panel, "", fill=tk.X, padx=8, pady=(2, 4))
        _sep(panel)

        _section(panel, "ДАТАСЕТ")
        _btn(panel, "Загрузить датасет", self.load, fill=tk.X, padx=6, pady=3)
        _btn(panel, "Другие примеры", self.resample, fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "датасет не загружен", fill=tk.X, padx=8,
                           pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.plot = PlotPanel(right, "gui_dataset_distribution", figsize=(7, 2.6))
        self.plot.pack(fill=tk.X, padx=6, pady=6)

        _label(right, "примеры вырезок (по строке на класс)", fg=ACCENT,
               font=MONO_B, fill=tk.X, padx=8)
        self.view = ImageView(right, width=880, height=300)
        self.view.pack(padx=8, pady=6)
        self._show_path()

    def _show_path(self):
        """
        Какая папка выбрана сейчас — видно до всякой загрузки.

        Заголовок окна обновляется здесь же. Он ставится при старте, и без
        этого после смены датасета продолжал бы показывать ПРЕЖНИЙ набор
        классов — то есть врал бы ровно о том, от чего зависит смысл всех
        остальных чисел на экране.
        """
        self.path_info.config(
            text=f"{os.path.basename(os.path.abspath(config.DATASET_DIR))}\n"
                 f"{config.N_CLASSES} кл.: " + ', '.join(config.CLASS_NAMES))
        self.root.title(f"EML Classifier — {config.dataset_summary()}")

    def choose_folder(self):
        """
        Выбрать папку, подпапки которой станут классами.

        Состав папки показывается ДО применения: смена датасета меняет число
        классов, а значит делает несовместимыми уже обученные модели. Узнать
        об этом лучше из диалога, чем из отказа загрузить оракула.
        """
        import data

        path = filedialog.askdirectory(
            title="Папка с подпапками-классами", mustexist=True,
            initialdir=os.path.abspath(config.DATASET_DIR)
                       if os.path.isdir(config.DATASET_DIR) else None)
        if not path:
            return

        info = data.describe_root(path)
        if not info['classes']:
            messagebox.showerror(
                "Не подходит",
                f"В папке {path} нет подпапок с картинками.\n\n"
                f"Ожидается:\n  <папка>/ИМЯ_КЛАССА/*.bmp|png|jpg\n\n"
                f"Каждая подпапка — класс, её имя — имя класса.")
            return

        listing = '\n'.join(f"    {name:<20} {info['counts'][name]:>6}"
                            for name in info['classes'])
        same = info['classes'] == list(config.CLASS_NAMES)
        warning = "" if same else (
            "\n\nНАБОР КЛАССОВ ДРУГОЙ. Уже обученные оракулы и формулы\n"
            "к нему не подойдут — их придётся обучить заново.")

        # JPEG своим декодером не читается. Сказать об этом здесь дешевле, чем
        # уронить загрузку датасета через пять минут после выбора.
        import imaging
        if ({'.jpg', '.jpeg'} & set(info['exts'])
                and imaging.backend_name() == 'нет'):
            warning += ("\n\nВ папке есть JPEG, а декодера для него нет.\n"
                        "Поставьте opencv-python или Pillow — иначе загрузка\n"
                        "датасета остановится на первом же .jpg.")
        if not messagebox.askokcancel(
                "Выбрать этот датасет?",
                f"{path}\n\nклассов: {len(info['classes'])}, "
                f"картинок: {info['n_files']}\n"
                f"форматы: {', '.join(info['exts'])}\n"
                f"размеры: {', '.join(info['sizes'])}"
                f" (приводятся к {config.IMG_SIZE}x{config.IMG_SIZE})\n\n"
                f"{listing}{warning}"):
            return

        config.use_dataset(path, info['classes'])
        config.save_dataset_choice()
        self.images = self.labels = None
        self._show_path()
        self.info.config(text="датасет выбран,\nнажмите «Загрузить датасет»")
        self.plot.clear()
        self.plot.draw()

    def load(self):
        import data
        self.images, self.labels, _ = data.load_dataset(verbose=False)
        counts = np.bincount(self.labels, minlength=config.N_CLASSES)
        train, val, test = data.stratified_split(self.labels)
        self.info.config(
            text=f"всего: {len(self.labels)}\n"
                 f"train: {len(train)}\nval:   {len(val)}\ntest:  {len(test)}\n\n"
                 + "\n".join(f"{c:<7} {counts[i]:>5}"
                             for i, c in enumerate(config.CLASS_NAMES))
                 + f"\n\nдисбаланс {counts.max() / counts.min():.1f} : 1")
        self._draw_distribution(counts)
        self.resample()

    def _draw_distribution(self, counts):
        self.plot.clear()
        ax = self.plot.axes(111)
        colors = [np.array(vision.class_color(i)) / 255 for i in range(config.N_CLASSES)]
        ax.bar(config.CLASS_NAMES, counts, color=colors)
        for i, v in enumerate(counts):
            ax.text(i, v, str(v), ha='center', va='bottom', color=FG, fontsize=8)
        ax.set_title("картинок по классам")
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


# ── Вкладка: Признаки ─────────────────────────────────────────────────────────

class FeaturesTab:
    """F-статистика признаков и их распределения по классам."""

    def __init__(self, root: tk.Tk, parent: tk.Frame):
        self.root = root
        self.feats = self.labels = None

        panel = tk.Frame(parent, bg=BG2, width=210)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "ПРИЗНАКИ")
        _btn(panel, "Посчитать признаки", self.compute, fill=tk.X, padx=6, pady=3)
        _sep(panel)
        _label(panel, "признак для разбора:", fill=tk.X, padx=8)
        self.choice = tk.StringVar(value=config.FEATURE_NAMES[0])
        self.combo = ttk.Combobox(panel, textvariable=self.choice, font=("Consolas", 8),
                                  values=config.FEATURE_NAMES, state='readonly')
        self.combo.pack(fill=tk.X, padx=6, pady=4)
        self.combo.bind('<<ComboboxSelected>>', lambda e: self._draw_detail())
        _sep(panel)
        self.info = _label(panel, "не посчитаны", fill=tk.X, padx=8, pady=4)

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
            text=f"признаков: {config.N_FEATURES}\n\nсильнейшие:\n"
                 + "\n".join(f"{config.FEATURE_NAMES[i][:15]:<15}{self.fscore[i]:6.0f}"
                             for i in order[:6])
                 + "\n\nслабейшие:\n"
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
        ax.set_title("F-статистика: насколько признак разделяет классы")
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


# ── Вкладка: Обучение ─────────────────────────────────────────────────────────

class TrainTab(WorkerTab):
    """Запуск обучения оракула и живые кривые."""

    REDRAW_INTERVAL = 0.4        # с; чаще перерисовывать кривые смысла нет

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        super().__init__(root, lock=lock, lock_name="обучение оракула")
        self.history = {'train_loss': [], 'val_acc': [], 'val_bal_acc': []}
        self._last_draw = 0.0

        panel = tk.Frame(parent, bg=BG2, width=210)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "ВЕТКА")
        self.branch = tk.StringVar(value='A')
        for value, text in (('A', "A — MLP на признаках"),
                            ('B', "B — CNN на пикселях")):
            tk.Radiobutton(panel, text=text, variable=self.branch, value=value,
                           bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                           activebackground=BG2, activeforeground=ACCENT,
                           anchor='w').pack(fill=tk.X, padx=6)

        params = _section(panel, "ПАРАМЕТРЫ")
        self.p_epochs = ParamRow(params, "эпох", config.TRAIN_EPOCHS, int)
        self.p_lr = ParamRow(params, "LR", config.TRAIN_LR, float)
        self.p_batch = ParamRow(params, "batch", config.TRAIN_BATCH_SIZE, int)
        # Имя копии в библиотеке. Пустое = время плюс метрика.
        self.p_name = ParamRow(params, "имя модели", "", str)

        _sep(panel)
        self.btn = _btn(panel, "Обучить", self.run, bg=GREEN, fg=BG,
                        fill=tk.X, padx=6, pady=3)
        self.btn_stop = _btn(panel, "Стоп", self.stop, bg=RED, fg=BG,
                             state=tk.DISABLED, fill=tk.X, padx=6, pady=3)
        self.btn_saved = _btn(panel, "Показать сохранённую", self.load_saved,
                              fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "готов", fill=tk.X, padx=8, pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot = PlotPanel(right, "gui_training", figsize=(8, 5.5))
        self.plot.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        if lock is not None:
            lock.subscribe(self._on_lock)

    def _on_lock(self, owner):
        """Пока работает другая вкладка — запуск отсюда недоступен."""
        foreign = owner is not None and owner != self.lock_name
        state = tk.DISABLED if foreign else tk.NORMAL
        self.btn_saved.config(state=state)
        if foreign:
            self.btn.config(state=tk.DISABLED)
        elif owner is None:
            self.btn.config(state=tk.NORMAL, text="Обучить")

    def run(self):
        branch = self.branch.get()
        epochs = self.p_epochs.get(config.TRAIN_EPOCHS)
        # LR и batch уезжают ПАРАМЕТРАМИ прогона. Раньше они писались в
        # config.TRAIN_LR / config.TRAIN_BATCH_SIZE и оставались там до конца
        # сессии, меняя поведение и соседних вкладок, и последующих запусков.
        lr = self.p_lr.get(config.TRAIN_LR)
        batch = self.p_batch.get(config.TRAIN_BATCH_SIZE)
        name = (self.p_name.get('') or '').strip() or None
        self.history = {'train_loss': [], 'val_acc': [], 'val_bal_acc': []}
        self._last_draw = 0.0
        if self.start(self._work, branch, epochs, lr, batch, name):
            self.btn.config(state=tk.DISABLED, text="Обучение...")
            self.btn_stop.config(state=tk.NORMAL)
            self.info.config(text=f"ветка {branch}: обучение...")

    def stop(self):
        self.request_stop()
        self.btn_stop.config(state=tk.DISABLED)
        self.info.config(text="останавливаю после эпохи...")

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
            messagebox.showinfo("Нет модели",
                                f"Оракул ветки {self.branch.get()} не обучен.")
            return
        self.on_event({'type': 'trained', 'history': meta.get('history', {}),
                       'meta': meta})

    def on_event(self, event):
        if event['type'] == 'epoch':
            self.history = event['history']
            self.info.config(
                text=f"эпоха {event['epoch'] + 1}/{event['epochs']}\n"
                     f"loss: {event['train_loss']:.4f}\n"
                     f"val bal: {event['val_bal_acc']:.4f}\n"
                     f"val acc: {event['val_acc']:.4f}\n"
                     f"лучшая:  {event['best_val_bal_acc']:.4f}\n"
                     f"LR: {event['lr']:.2e}")
            now = time.perf_counter()
            if now - self._last_draw > self.REDRAW_INTERVAL:
                self._last_draw = now
                self._draw()
        elif event['type'] == 'trained':
            self.history = event.get('history') or {}
            meta = event['meta']
            note = "\n(остановлено вручную)" if meta.get('stopped_early') else ""
            self.info.config(
                text=f"{meta.get('model', '?')}\n"
                     f"параметров: {meta.get('params', 0):,}\n"
                     f"val bal:  {meta.get('val_bal_acc', 0):.4f}\n"
                     f"val acc:  {meta.get('val_acc', 0):.4f}\n"
                     f"test bal: {meta.get('test_bal_acc', 0):.4f}\n"
                     f"test acc: {meta.get('test_acc', 0):.4f}\n"
                     f"эпох: {meta.get('epochs_run', 0)}\n"
                     f"время: {meta.get('seconds', 0):.0f} c{note}\n\n"
                     f"в библиотеке:\n{meta.get('library_name', '—')}")
            self._draw()
        elif event['type'] == 'error':
            messagebox.showerror("Ошибка обучения", event['text'])
            self.info.config(text="ошибка")
        elif event['type'] == 'done':
            self.btn.config(state=tk.NORMAL, text="Обучить")
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
        # Обе кривые: отбор идёт по сбалансированной, а разрыв между ними
        # показывает, насколько модель держится за многочисленный класс.
        ax2.plot(acc, color=FG2, alpha=0.8, label="val accuracy")
        curve = bal or acc
        if bal:
            ax2.plot(bal, color=GREEN, label="val bal. accuracy")
        best = int(np.argmax(curve))
        ax2.axvline(best, color=YELLOW, ls='--', alpha=0.7)
        ax2.set_title(f"{'val bal.accuracy' if bal else 'val accuracy'} "
                      f"(лучшая {curve[best]:.4f} на эпохе {best})")
        ax2.set_xlabel("эпоха")
        leg = ax2.legend(fontsize=7, facecolor=BG2, edgecolor=BG3)
        for text in leg.get_texts():
            text.set_color(FG2)
        self.plot.fig.tight_layout()
        self.plot.draw()


# ── Вкладка: Дистилляция ──────────────────────────────────────────────────────

class DistillTab(WorkerTab):
    """Запуск GA-дистилляции с живыми кривыми обеих фаз."""

    REDRAW_INTERVAL = 0.5        # с; кривые GA дёргаются каждое поколение

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        super().__init__(root, lock=lock, lock_name="дистилляция")
        self.data_hist = {}
        self.acc_hist = []
        self.single_hist = []
        self.island_hist = []
        self._last_draw = 0.0

        panel = tk.Frame(parent, bg=BG2, width=215)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "ВЕТКА")
        self.branch = tk.StringVar(value='A')
        for value, text in (('A', "A — по формуле на класс"),
                            ('B', "B — эмбеддинг CNN"),
                            ('C', "C — ОДНА формула")):
            tk.Radiobutton(panel, text=text, variable=self.branch, value=value,
                           bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                           activebackground=BG2, activeforeground=ACCENT,
                           anchor='w', command=self._on_branch
                           ).pack(fill=tk.X, padx=6)

        # Дефолты берутся из config, а не вписаны числами: раньше здесь стояли
        # 300/400 против 600/600 в config, и запуск из окна тихо считал не то
        # же, что `python main.py distill` — а сравнивали потом их результаты.
        params = _section(panel, "ЭВОЛЮЦИЯ")
        self.p_gen = ParamRow(params, "поколений", config.EML_GENERATIONS, int)
        self.p_pop = ParamRow(params, "популяция", config.EML_POPULATION, int)
        self.p_seed = ParamRow(params, "сид", 0, int)
        self.p_name = ParamRow(params, "имя модели", "", str)
        _label(panel, "штраф за сложность:", fill=tk.X, padx=8, pady=(6, 0))
        self.penalty = tk.StringVar(value=config.DEPTH_PENALTY_DEFAULT)
        ttk.Combobox(panel, textvariable=self.penalty, font=("Consolas", 8),
                     values=list(config.DEPTH_PENALTIES), state='readonly'
                     ).pack(fill=tk.X, padx=6, pady=3)

        self.acc_phase = tk.BooleanVar(value=True)
        self.chk_acc = tk.Checkbutton(
            panel, text="ACC-фаза", variable=self.acc_phase, bg=BG2, fg=FG,
            selectcolor=BG3, font=("Consolas", 8), activebackground=BG2,
            anchor='w')
        self.chk_acc.pack(fill=tk.X, padx=6, pady=4)

        # Ветка C идёт часами, поэтому её ручки — про то, как расходовать это
        # время: сколько независимых популяций и сколько процессов под них.
        self.c_frame = tk.Frame(panel, bg=BG2)
        self.p_islands = ParamRow(self.c_frame, "островов", 0, int)
        self.p_workers = ParamRow(self.c_frame, "процессов", 0, int)
        self.resume = tk.BooleanVar(value=True)
        tk.Checkbutton(self.c_frame, text="продолжить прогон",
                       variable=self.resume, bg=BG2, fg=FG, selectcolor=BG3,
                       font=("Consolas", 8), activebackground=BG2,
                       anchor='w').pack(fill=tk.X, padx=6, pady=2)
        _label(self.c_frame, "0 = по числу ядер", fg=FG2, fill=tk.X, padx=8)
        _sep(panel)
        self.btn = _btn(panel, "Дистиллировать", self.run, bg=PURPLE, fg=BG,
                        fill=tk.X, padx=6, pady=3)
        self.btn_stop = _btn(panel, "Стоп", self.stop, bg=RED, fg=BG,
                             state=tk.DISABLED, fill=tk.X, padx=6, pady=3)
        self.btn_saved = _btn(panel, "Показать сохранённые", self.load_saved,
                              fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "готов", fill=tk.X, padx=8, pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot = PlotPanel(right, "gui_distill", figsize=(8, 3.6))
        self.plot.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        _label(right, "формулы", fg=ACCENT, font=MONO_B, fill=tk.X, padx=8)
        self.text = tk.Text(right, bg=BG2, fg=FG, font=("Consolas", 8),
                            relief=tk.FLAT, height=11, wrap=tk.WORD,
                            insertbackground=FG)
        self.text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))

        if lock is not None:
            lock.subscribe(self._on_lock)
        self._on_branch()

    def _on_branch(self):
        """
        Показать ручки той ветки, которая выбрана.

        У ветки C нет ACC-фазы (она одна целиком), зато есть острова, процессы
        и продолжение прогона. Оставить на экране чужие поля значило бы дать
        покрутить то, что ни на что не влияет.
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
            self.btn.config(state=tk.NORMAL, text="Дистиллировать")

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
        self.btn.config(state=tk.DISABLED, text="Эволюция...")
        self.btn_stop.config(state=tk.NORMAL)
        self.info.config(text="дистилляция...")
        self.text.delete('1.0', tk.END)

    def stop(self):
        self.request_stop()
        self.btn_stop.config(state=tk.DISABLED)
        self.info.config(text="останавливаю после поколения...")

    def _work(self, branch, generations, population, penalty, acc_phase, seed,
              name=None):
        import pipeline

        # Прогресс приходит каждое поколение — это тысячи событий за прогон.
        # В очередь кладём не чаще чем раз в 0.2 с: интерфейсу больше не нужно,
        # а забивать очередь списками истории незачем.
        last = [0.0]

        def progress(payload):
            now = time.perf_counter()
            final = payload['gen'] == payload['generations'] - 1
            if final or now - last[0] > 0.2:
                last[0] = now
                self.queue.put({'type': 'progress', **payload})

        if branch == 'C':
            # Ветка C считает в отдельных ПРОЦЕССАХ (см. eml_single). Из потока
            # интерфейса это законно: дочерние процессы порождаются через
            # spawn, а точка входа проекта закрыта проверкой __main__.
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
            messagebox.showinfo("Нет формул", str(exc))
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
            note = f"\n\nв библиотеке:\n{meta.get('library_name', '—')}"
            if meta.get('stopped'):
                note += "\n\n(прервано вручную —\nне стало активной)"
            elif saved and not saved.get('promoted'):
                note += f"\n\nактивная не тронута:\n{saved.get('reason', '')}"
            self.info.config(
                text=f"узлов: {meta.get('total_nodes', 0)}\n"
                     f"глубина: {meta.get('max_depth', 0)}\n"
                     f"входов: {meta.get('n_inputs_used', 0)}"
                     f"/{meta.get('n_inputs_available', 0)}\n\n"
                     f"test EML:    {metrics.get('eml_bal_acc', 0):.4f}\n"
                     f"test оракул: {metrics.get('oracle_bal_acc', 0):.4f}\n"
                     f"доля: {metrics.get('eml_bal_acc', 0) / oracle_bal * 100:.1f}%\n"
                     f"fidelity: {metrics.get('fidelity', 0):.4f}{note}")
            self.text.delete('1.0', tk.END)
            if meta.get('class_layout'):
                # Ветка C: формула одна, а главный результат — куда она сама
                # расставила классы на своей оси.
                self.text.insert(tk.END, f"f =\n  {event['formulas'][0]}\n\n")
                self.text.insert(
                    tk.END, f"участков оси: {meta.get('n_intervals', '?')}\n"
                            f"{meta['class_layout']}\n")
            else:
                for c, formula in enumerate(event['formulas']):
                    name = (config.CLASS_NAMES[c] if c < config.N_CLASSES
                            else f'C{c}')
                    self.text.insert(tk.END, f"f_{name} =\n  {formula}\n\n")
            if event.get('result'):
                self._draw_result(event['result'])
        elif event['type'] == 'error':
            messagebox.showerror("Ошибка дистилляции", event['text'])
            self.info.config(text="ошибка")
        elif event['type'] == 'done':
            self.btn.config(state=tk.NORMAL, text="Дистиллировать")
            self.btn_stop.config(state=tk.DISABLED)

    def _on_progress(self, event):
        """Живые кривые: DATA-фаза копит по кривой на класс, ACC — одну."""
        if event['phase'] == 'single':
            self.single_hist = event['history']
            self.island_hist = event.get('islands') or []
            spread = (max(self.island_hist) - min(self.island_hist)
                      if len(self.island_hist) > 1 else 0.0)
            self.info.config(
                text=f"ветка C\n"
                     f"поколение {event['gen']}/{event['generations']}\n"
                     f"fitness: {event['best']:.4f}\n"
                     f"узлов: {event['nodes']}, глубина {event['depth']}\n"
                     f"участков: {event['intervals']}\n"
                     f"островов: {len(self.island_hist)}\n"
                     f"разброс: {spread:.4f}")
        elif event['phase'] == 'data':
            self.data_hist[event['class_idx']] = event['history']
            variant = (f"вариант {event['variant'] + 1}/{event['variants']}  "
                       if 'variant' in event else "")
            self.info.config(
                text=f"{variant}DATA-фаза\n"
                     f"{event['class_name']} "
                     f"({event['class_idx'] + 1}/{event['n_classes']})\n"
                     f"поколение {event['gen'] + 1}/{event['generations']}\n"
                     f"fitness: {event['best']:.3f}\n"
                     f"узлов: {event['nodes']}, глубина {event['depth']}")
        else:
            self.acc_hist = event['history']
            self.info.config(
                text=f"ACC-фаза\n"
                     f"поколение {event['gen'] + 1}/{event['generations']}\n"
                     f"bal.accuracy: {event['best']:.4f}\n"
                     f"узлов: {event['nodes']}")

        now = time.perf_counter()
        if now - self._last_draw > self.REDRAW_INTERVAL:
            self._last_draw = now
            self._draw(self.data_hist, self.acc_hist)

    def _draw_result(self, result):
        """Итоговые кривые из готового прогона (или из «Показать сохранённые»)."""
        if 'model' in result:                     # ветка C: одна кривая
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
            ax.set_title("ветка C: fitness (сбаланс. accuracy − штрафы)")
            ax.set_xlabel("раунд обмена между островами")
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
        ax.set_title("DATA-фаза: fitness регрессии")
        ax.set_xlabel("поколение")
        if data_hist:
            leg = ax.legend(fontsize=7, facecolor=BG2, edgecolor=BG3)
            for text in leg.get_texts():
                text.set_color(FG2)

        if has_acc:
            ax2 = self.plot.axes(1, 2, 2)
            ax2.plot(acc_hist, color=GREEN)
            ax2.set_title("ACC-фаза: сбалансированная accuracy")
            ax2.set_xlabel("поколение")
        self.plot.fig.tight_layout()
        self.plot.draw()


# ── Вкладка: Модели ───────────────────────────────────────────────────────────

ACTIVE_CHOICE = "— активная —"


def _lib_branch(kind: str, branch: str) -> str:
    """
    Под какой веткой искать модель этого вида в библиотеке.

    Своего оракула у ветки C нет: она учится у того же MLP, что и ветка A, и
    в её пространстве признаков. Поэтому запрос «оракул ветки C» надо
    переадресовать ветке A, иначе списки моделей окажутся пустыми, а
    сравнение «формула против сети» — недоступным именно там, где оно нужнее
    всего.
    """
    return 'A' if kind == 'oracle' and branch.upper() == 'C' else branch


def _model_choices(kind: str, branch: str) -> list[str]:
    """
    Список для выпадающего меню: «активная» плюс модели библиотеки.

    Первым пунктом всегда активная модель — в большинстве случаев нужна
    именно она, а конкретная версия выбирается, когда сравнивают две.
    """
    import storage
    return [ACTIVE_CHOICE] + [e.label() for e in storage.list_models(kind, branch)]


def _resolve_choice(choice: str, kind: str, branch: str) -> str | None:
    """Пункт меню -> путь к файлу модели (None = активный слот)."""
    import storage
    if not choice or choice == ACTIVE_CHOICE:
        return None
    for entry in storage.list_models(kind, branch):
        if entry.label() == choice:
            return entry.path
    return None                      # модель удалили — молча берём активную


class ModelsTab:
    """
    Библиотека моделей: что сохранено, что активно, что лучшее.

    Обучение и дистилляция стоят от секунд до часа, а разброс GA по сидам
    больше разницы между правками метода — поэтому «какую модель мы сейчас
    меряем» это не мелочь интерфейса. Здесь видно все сохранённые модели с их
    характеристиками, и любую можно сделать активной.
    """

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        self.root = root
        self.entries: list = []

        panel = tk.Frame(parent, bg=BG2, width=250)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "БИБЛИОТЕКА")
        _btn(panel, "Обновить список", self.refresh, fill=tk.X, padx=6, pady=3)
        self.btn_activate = _btn(panel, "Сделать активной", self.activate,
                                 bg=GREEN, fg=BG, fill=tk.X, padx=6, pady=3)
        self.btn_rename = _btn(panel, "Переименовать...", self.rename,
                               fill=tk.X, padx=6, pady=3)
        self.btn_delete = _btn(panel, "Удалить...", self.delete, bg=RED, fg=BG,
                               fill=tk.X, padx=6, pady=3)
        _sep(panel)
        _label(panel, "★ лучшая по val\n• активная сейчас", fg=FG2,
               fill=tk.X, padx=8, pady=(0, 4))
        _sep(panel)
        self.info = _label(panel, "выберите модель в списке", fill=tk.X,
                           padx=8, pady=4)

        right = tk.Frame(parent, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _label(right, "сохранённые модели", fg=ACCENT, font=MONO_B,
               fill=tk.X, padx=8, pady=(8, 2))

        columns = ('marks', 'kind', 'branch', 'name', 'val', 'test', 'size',
                   'when')
        self.tree = ttk.Treeview(right, columns=columns, show='headings',
                                 selectmode='browse', height=18)
        for column, title, width, anchor in (
                ('marks', '', 40, 'center'),
                ('kind', 'вид', 80, 'w'),
                ('branch', 'ветка', 55, 'center'),
                ('name', 'имя', 300, 'w'),
                ('val', 'val bal', 80, 'e'),
                ('test', 'test bal', 80, 'e'),
                ('size', 'сложность', 110, 'e'),
                ('when', 'сохранено', 120, 'w')):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, anchor=anchor)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.tree.bind('<<TreeviewSelect>>', lambda e: self._on_select())
        self.tree.tag_configure('best', foreground=GREEN)
        self.tree.tag_configure('active', foreground=ACCENT)

        _label(right, "Активная модель — та, которую берут экспорт в C, "
                      "verify-c, bench и вкладки теста\nпо умолчанию. Прогоны "
                      "попадают в библиотеку сами; лучшая выбирается по val.",
               fg=FG2, fill=tk.X, padx=10, pady=(0, 8))

        if lock is not None:
            lock.subscribe(self._on_lock)
        self.refresh()

    def _on_lock(self, owner):
        """Пока идёт обучение/дистилляция, файлы моделей не трогаем."""
        state = tk.DISABLED if owner is not None else tk.NORMAL
        for button in (self.btn_activate, self.btn_rename, self.btn_delete):
            button.config(state=state)

    def refresh(self):
        import storage

        self.tree.delete(*self.tree.get_children())
        self.entries = storage.list_models()
        for i, entry in enumerate(self.entries):
            marks = ('★' if entry.is_best else '') + ('•' if entry.is_active else '')
            unit = 'пар.' if entry.kind == 'oracle' else 'узл.'
            tags = []
            if entry.is_active:
                tags.append('active')
            elif entry.is_best:
                tags.append('best')
            self.tree.insert(
                '', tk.END, iid=str(i), tags=tags, values=(
                    marks,
                    'оракул' if entry.kind == 'oracle' else 'формулы',
                    entry.branch,
                    entry.name,
                    f"{entry.val_score:.4f}" if entry.val_score is not None else '—',
                    f"{entry.test_score:.4f}" if entry.test_score is not None else '—',
                    f"{entry.size} {unit}" if entry.size is not None else '—',
                    time.strftime('%d.%m %H:%M', time.localtime(entry.saved_at))))
        if not self.entries:
            self.info.config(text="библиотека пуста.\n\nМодели попадают сюда\n"
                                  "сами при обучении\nи дистилляции.")

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
            messagebox.showinfo("Не выбрано", "Выберите модель в списке.")
            return
        if entry.is_active:
            messagebox.showinfo("Уже активна",
                                f"«{entry.name}» и так активна.")
            return
        storage.activate(entry)
        self.refresh()
        self.info.config(text=f"активной стала\n«{entry.name}»")

    def rename(self):
        import storage
        from tkinter import simpledialog

        entry = self._selected()
        if entry is None:
            messagebox.showinfo("Не выбрано", "Выберите модель в списке.")
            return
        new_name = simpledialog.askstring("Переименовать",
                                          "Новое имя модели:",
                                          initialvalue=entry.name,
                                          parent=self.root)
        if not new_name or new_name == entry.name:
            return
        renamed = storage.rename_model(entry, new_name)
        self.refresh()
        self.info.config(text=f"новое имя:\n«{renamed.name}»")

    def delete(self):
        import storage

        entry = self._selected()
        if entry is None:
            messagebox.showinfo("Не выбрано", "Выберите модель в списке.")
            return
        note = ("\n\nЭта модель сейчас активна. Активный слот останется на "
                "месте, из библиотеки исчезнет только копия."
                if entry.is_active else "")
        if not messagebox.askyesno("Удалить модель",
                                   f"Удалить «{entry.name}» безвозвратно?{note}"):
            return
        storage.delete_model(entry)
        self.refresh()
        self.info.config(text="удалено")


# ── Вкладка: Тест · Фото ──────────────────────────────────────────────────────

class PhotoTab:
    """
    Классификация одной картинки: ответ формул рядом с ответом оракула.

    Смысл вкладки — не «показать предсказание», а сравнить два движка на одном
    и том же кадре: где они согласны, где расходятся и насколько уверены.
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

        _section(panel, "ВЕТКА")
        self.branch = tk.StringVar(value='A')
        for value, text in (('A', "ветка A"), ('B', "ветка B"),
                            ('C', "ветка C (одна формула)")):
            tk.Radiobutton(panel, text=text, variable=self.branch,
                           value=value, bg=BG2, fg=FG, selectcolor=BG3,
                           font=("Consolas", 8), activebackground=BG2,
                           anchor='w',
                           command=self._on_branch).pack(fill=tk.X, padx=6)

        # Выбор конкретных моделей. Сравнивать «формулы против оракула» и
        # «эту версию формул против той» — одна и та же операция, поэтому
        # движки выбираются независимо.
        _section(panel, "МОДЕЛИ")
        _label(panel, "оракул:", fill=tk.X, padx=8)
        self.pick_oracle = tk.StringVar(value=ACTIVE_CHOICE)
        self.combo_oracle = ttk.Combobox(panel, textvariable=self.pick_oracle,
                                         font=("Consolas", 7), state='readonly')
        self.combo_oracle.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.combo_oracle.bind('<<ComboboxSelected>>', lambda e: self.classify())

        _label(panel, "формулы:", fill=tk.X, padx=8)
        self.pick_eml = tk.StringVar(value=ACTIVE_CHOICE)
        self.combo_eml = ttk.Combobox(panel, textvariable=self.pick_eml,
                                      font=("Consolas", 7), state='readonly')
        self.combo_eml.pack(fill=tk.X, padx=6, pady=(0, 4))
        self.combo_eml.bind('<<ComboboxSelected>>', lambda e: self.classify())
        _btn(panel, "Обновить список моделей", self.refresh_models,
             fill=tk.X, padx=6, pady=3)

        _section(panel, "ИСТОЧНИК")
        self.btn_random = _btn(panel, "Случайная из датасета", self.random_sample,
                               fill=tk.X, padx=6, pady=3)
        _label(panel, "класс:", fill=tk.X, padx=8, pady=(4, 0))
        self.class_filter = tk.StringVar(value="любой")
        ttk.Combobox(panel, textvariable=self.class_filter, font=("Consolas", 8),
                     values=["любой"] + config.CLASS_NAMES, state='readonly'
                     ).pack(fill=tk.X, padx=6, pady=3)
        self.btn_open = _btn(panel, "Открыть файл...", self.open_file,
                             fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "картинка не выбрана", fill=tk.X, padx=8,
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
        """Пока идёт обучение/дистилляция, модели с диска не читаем."""
        state = tk.DISABLED if owner is not None else tk.NORMAL
        self.btn_random.config(state=state)
        self.btn_open.config(state=state)
        if owner is None:
            # Прогон закончился — в библиотеке появилась новая модель.
            self.refresh_models()

    def refresh_models(self):
        """Перечитать библиотеку в оба выпадающих списка."""
        branch = self.branch.get()
        for combo, var, kind in ((self.combo_oracle, self.pick_oracle, 'oracle'),
                                 (self.combo_eml, self.pick_eml, 'eml')):
            choices = _model_choices(kind, _lib_branch(kind, branch))
            combo['values'] = choices
            if var.get() not in choices:
                var.set(ACTIVE_CHOICE)

    def _on_branch(self):
        """Смена ветки: у неё свои модели, список надо пересобрать."""
        self.refresh_models()
        self.classify()

    def _predictor(self, engine: str) -> vision.Predictor:
        """
        Модели кэшируются: перезагружать их на каждый клик бессмысленно.

        Но кэш обязан замечать переобучение. Раньше ключом была пара
        (ветка, движок), и после обучения или дистилляции в соседней вкладке
        фото-тест продолжал молча показывать ответы СТАРОЙ модели — то есть
        сравнивал не то, что лежит на диске. Поэтому в ключ входит ещё и время
        правки файла: чекпоинт переписали — ключ другой — модель грузится
        заново.
        """
        import storage

        branch = self.branch.get()
        kind = 'oracle' if engine == 'oracle' else 'eml'
        lib_branch = _lib_branch(kind, branch)
        choice = (self.pick_oracle if engine == 'oracle' else self.pick_eml).get()
        chosen = _resolve_choice(choice, kind, lib_branch)

        # Ветка B на формулах зависит от ДВУХ файлов: вход формул — эмбеддинг
        # CNN, поэтому переобучение оракула B меняет ответ EML-движка ровно так
        # же, как новая дистилляция.
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
            # Держать все версии подряд незачем — интересна только свежая.
            self._predictors = {k: v for k, v in self._predictors.items()
                                if k[:2] != (branch, engine)}
            self._predictors[key] = predictor
        return self._predictors[key]

    def random_sample(self):
        import data
        if self.images is None:
            self.images, self.labels, _ = data.load_dataset(verbose=False)
        pool = np.arange(len(self.labels))
        if self.class_filter.get() != "любой":
            c = config.CLASS_NAMES.index(self.class_filter.get())
            pool = np.flatnonzero(self.labels == c)
        i = int(np.random.default_rng().choice(pool))
        self.crop = self.images[i]
        self.true_label = int(self.labels[i])
        self.classify()

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Выберите картинку",
            filetypes=[("Изображения", "*.bmp *.png *.jpg *.jpeg"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            frame = _read_image(path)
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("Не удалось открыть", str(exc))
            return
        # Файл может быть любого размера, а модель обучена на 64x64 —
        # берём центральное окно, а не растягиваем весь кадр: растяжение
        # исказило бы масштаб текстуры, на который опираются признаки.
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
                # Битый JSON формул, несовпадение размерности после смены
                # набора признаков, повреждённый чекпоинт — всё это раньше
                # летело необработанным исключением в главный поток и роняло
                # клик по кнопке молча, без единого слова в окне.
                messagebox.showerror(
                    "Не удалось загрузить модель",
                    f"{'формулы' if engine == 'eml' else 'оракул'}, "
                    f"ветка {self.branch.get()}:\n{exc}")
                continue
            preds, scores, conf = predictor.predict(self.crop[None])
            results[engine] = (int(preds[0]), scores[0], float(conf[0]),
                               predictor.label(), predictor.conf_is_probability,
                               predictor)
        if not results:
            messagebox.showinfo("Нет моделей",
                                f"Для ветки {self.branch.get()} нет обученной "
                                f"модели или формул.")
            return

        lines = []
        if self.true_label is not None:
            lines.append(f"истина:  {config.CLASS_NAMES[self.true_label]}")
        for engine, (pred, _, conf, label, is_prob, predictor) in results.items():
            mark = ''
            if self.true_label is not None:
                mark = '  верно' if pred == self.true_label else '  ОШИБКА'
            # У оракула это вероятность после softmax, у формул — просто
            # оценка: их выходы никто не калибровал как вероятности.
            # Одинаково подписывать два разных числа нельзя, и как именно
            # подписывать — знает сам движок, а не эта вкладка.
            suffix = (f"{conf * 100:4.0f}%" if is_prob
                      else f"{predictor.conf_label()} {conf:.2f}")
            lines.append(f"{'формулы' if engine == 'eml' else 'оракул':<8} "
                         f"{config.CLASS_NAMES[pred]:<7} {suffix}{mark}")
        if len(results) == 2:
            agree = results['eml'][0] == results['oracle'][0]
            lines.append("")
            lines.append("движки согласны" if agree else "движки РАСХОДЯТСЯ")
        lines.append("")
        lines.append("% оракула — вероятность,")
        eml_predictor = results.get('eml', (None,) * 6)[5]
        if eml_predictor is not None and eml_predictor.is_single:
            lines.append("запас — доля участка оси до")
            lines.append("ближайшего порога, 0..1")
        elif eml_predictor is not None:
            lines.append("оценка формул — не вероятность")
            lines.append(f"(минимум для {config.N_CLASSES} классов "
                         f"{eml_predictor.conf_floor:.2f})")
        self.verdict.config(text="\n".join(lines))
        self.info.config(text=f"кадр {self.crop.shape[1]}x{self.crop.shape[0]}\n"
                              f"ветка {self.branch.get()}")
        self._draw_scores(results)

    def _draw_scores(self, results):
        self.plot.clear()
        ax = self.plot.axes(111)
        width = 0.38
        x = np.arange(config.N_CLASSES)
        for k, item in enumerate(results.items()):
            engine, (_, scores, _, label, _is_prob, predictor) = item
            if predictor.is_single:
                # Ветка C: оценка класса — минус расстояние по оси выхода, и
                # у неё нет верхней шкалы. Softmax от таких чисел насыщается в
                # единицу почти всегда, поэтому столбик рисуется по-другому:
                # 1 у предсказанного класса, у остальных — насколько близко
                # они были, в долях от самого далёкого.
                worst = float(np.min(scores))
                probs = (1.0 - scores / worst) if worst < 0 else np.ones_like(scores)
            else:
                # Оценки движков живут в разных шкалах (логиты против
                # калиброванных выходов формул), поэтому сравнивать их можно
                # только после softmax — иначе столбики несопоставимы.
                probs = vision.softmax(scores[None])[0]
            ax.bar(x + (k - 0.5) * width, probs, width,
                   label=label, color=ACCENT if engine == 'eml' else PURPLE)
        ax.set_xticks(x)
        ax.set_xticklabels(config.CLASS_NAMES, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("оценки классов после softmax "
                     "(у формул это не вероятности)")
        leg = ax.legend(fontsize=7, facecolor=BG2, edgecolor=BG3)
        for text in leg.get_texts():
            text.set_color(FG2)
        self.plot.fig.tight_layout()
        self.plot.draw()


def _read_image(path: str) -> np.ndarray:
    """
    Прочитать картинку в RGB.

    BMP и PNG читаются своими парсерами (imaging.py) и не требуют ничего
    стороннего; JPEG — только через OpenCV или Pillow, если они установлены.
    """
    import imaging
    return imaging.read_image(path)


# ── Вкладка: Тест · Видео ─────────────────────────────────────────────────────

class VideoTab(WorkerTab):
    """
    Разметка видео в реальном времени.

    Кадр режется на окна 64x64 — ровно того размера, на котором обучалась
    модель, — каждое окно классифицируется, и поверх кадра ложится
    полупрозрачная заливка цветом класса.

    Декодирование и инференс идут в рабочем потоке, готовые кадры уходят в
    очередь, а рисует их главный поток. Иначе интерфейс замирал бы на время
    обработки каждого кадра.
    """

    MAX_QUEUE = 4        # больше кадров про запас копить незачем: видео живое

    # Потолок буфера записи. Кадр 900x560 RGB это ~1.5 МБ, 12 кадров/с — почти
    # 18 МБ в секунду. Без потолка десять минут записи это гигабайты в списке
    # Python, и приложение падает по памяти на ровном месте. 600 кадров ~ минута
    # видео и меньше гигабайта; дальше запись останавливается с предупреждением
    # в панели, а не молча съедает машину.
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
        self._running_model = None        # (ветка, движок) текущего источника
        self._source = None               # чем перезапускать при смене модели
        # Снимок настроек оверлея обычными числами.
        #
        # Читать tkinter-переменные из рабочего потока НЕЛЬЗЯ: Tk не
        # потокобезопасен и отвечает «main thread is not in main loop». Причём
        # падает не всегда, а как повезёт, поэтому ловится такое плохо.
        # Поэтому значения снимаются в главном потоке (_refresh_settings,
        # вызывается из _tick), а поток видео читает только этот словарь.
        self._live = {'stride': vision.DEFAULT_STRIDE, 'alpha': 0.35,
                      'conf': 0.0}

        panel = tk.Frame(parent, bg=BG2, width=215)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        _section(panel, "ИСТОЧНИК")
        self.src_buttons = [
            _btn(panel, "Видеофайл...", self.open_file, fill=tk.X, padx=6, pady=3),
            _btn(panel, "Веб-камера", self.open_camera, fill=tk.X, padx=6, pady=3),
            _btn(panel, "Синтетическое из датасета", self.open_synthetic,
                 fill=tk.X, padx=6, pady=3),
        ]

        _section(panel, "МОДЕЛЬ")
        self.branch = tk.StringVar(value='A')
        self.engine = tk.StringVar(value='eml')
        # Переключение модели во время проигрывания перезапускает источник.
        # Раньше радиокнопки просто ничего не делали до следующего запуска:
        # пользователь выбирал «оракул», видел прежнюю разметку формул и
        # честно считал, что это ответ оракула.
        self.model_radios = []
        for value, text in (('A', "ветка A"), ('B', "ветка B"),
                            ('C', "ветка C")):
            self.model_radios.append(tk.Radiobutton(
                panel, text=text, variable=self.branch, value=value,
                bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                activebackground=BG2, anchor='w',
                command=self._on_model_change))
            self.model_radios[-1].pack(fill=tk.X, padx=6)
        for value, text in (('eml', "EML-формулы"), ('oracle', "оракул")):
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

        params = _section(panel, "ОВЕРЛЕЙ")
        self.p_stride = ParamRow(params, "шаг окна", vision.DEFAULT_STRIDE, int)
        self.p_alpha = ParamRow(params, "прозрачность", 0.35, float)
        self.p_conf = ParamRow(params, "порог увер.", 0.0, float)
        self.show_labels = tk.BooleanVar(value=True)
        tk.Checkbutton(panel, text="подписи классов", variable=self.show_labels,
                       bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                       activebackground=BG2, anchor='w').pack(fill=tk.X, padx=6,
                                                              pady=3)

        _sep(panel)
        self.btn_stop = _btn(panel, "Стоп", self.stop, bg=RED, fg=BG,
                             state=tk.DISABLED, fill=tk.X, padx=6, pady=3)
        _btn(panel, "Сохранить кадр", self.save_frame, fill=tk.X, padx=6, pady=3)
        self.rec_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="писать результат", variable=self.rec_var,
                       bg=BG2, fg=FG, selectcolor=BG3, font=("Consolas", 8),
                       activebackground=BG2, anchor='w').pack(fill=tk.X, padx=6)
        _btn(panel, "Сохранить видео", self.save_video, fill=tk.X, padx=6, pady=3)
        _sep(panel)
        self.info = _label(panel, "источник не выбран", fill=tk.X, padx=8, pady=4)

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
        """Пока идёт обучение/дистилляция, новый источник не запускаем."""
        state = tk.DISABLED if owner is not None else tk.NORMAL
        for button in self.src_buttons:
            button.config(state=state)
        if owner is None:
            self.refresh_models()

    def refresh_models(self):
        """Список моделей для текущей пары (ветка, движок)."""
        kind = 'oracle' if self.engine.get() == 'oracle' else 'eml'
        choices = _model_choices(kind, _lib_branch(kind, self.branch.get()))
        self.combo_model['values'] = choices
        if self.pick_model.get() not in choices:
            self.pick_model.set(ACTIVE_CHOICE)

    def _current_model(self) -> tuple[str, str, str | None]:
        """(ветка, движок, путь) — то, чем сейчас должна размечаться картинка."""
        branch, engine = self.branch.get(), self.engine.get()
        kind = 'oracle' if engine == 'oracle' else 'eml'
        return branch, engine, _resolve_choice(self.pick_model.get(), kind,
                                               _lib_branch(kind, branch))

    def _on_model_change(self):
        """Смена модели на ходу = перезапуск источника с новой моделью."""
        self.refresh_models()
        if self._running_model is None or not self.busy:
            return
        if self._running_model == self._current_model():
            return
        self._start_source(self._source)

    # ── Источники ─────────────────────────────────────────────────────────
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv"),
                       ("Все файлы", "*.*")])
        if path:
            self._start_source(path)

    def open_camera(self):
        self._start_source(0)

    def open_synthetic(self):
        """
        Видео, собранное из вырезок датасета.

        Нужно затем, чтобы видео-режим можно было проверить без аэросъёмки под
        рукой: у каждого окна здесь известна истинная метка, поэтому в углу
        показывается настоящая точность разметки, а не только красивые рамки.
        """
        self._start_source('synthetic')

    def _start_source(self, source):
        # Старый поток надо не просто попросить остановиться, но и дождаться:
        # иначе start() увидит его живым, решит, что занято, и смена источника
        # молча не сработает.
        self.request_stop()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        _drain(self.frames)
        self._refresh_settings()      # чтобы первый же кадр считался как надо
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
        self.info.config(text=f"источник: {source}\n"
                              f"модель: {branch}/{engine}\nзапуск...")

    def stop(self):
        self.request_stop()
        self._running_model = None
        self.btn_stop.config(state=tk.DISABLED)

    # ── Рабочий поток ─────────────────────────────────────────────────────
    def _work(self, source, branch, engine, path=None):
        predictor = vision.Predictor(branch=branch, engine=engine, path=path)
        video = None
        if source == 'synthetic':
            stream = self._synthetic_stream()
            fps_limit = 0.0
        else:
            video = vision.VideoSource(source)
            stream = self._video_stream(video)
            # Файл проигрывается в его собственном темпе, камера — как успеет.
            fps_limit = video.fps if not video.is_camera else 0.0

        try:
            for frame, truth in stream:
                if self.should_stop():
                    break
                t0 = time.perf_counter()
                settings = dict(self._live)      # снимок, сделанный главным потоком
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
                    # Интерфейс не успевает — кадр выбрасывается. Для живого
                    # видео это правильнее, чем копить отставание.
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
                video.rewind()                        # файл — по кругу
                frame = video.read()
                if frame is None:
                    break
            yield frame, None

    def _synthetic_stream(self):
        """Медленно проматывающаяся мозаика из вырезок с известными метками."""
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
            cursor += cols                            # сдвиг на строку за кадр
            frame = vision.mosaic(images[sel], rows, cols)
            yield frame, labels[sel]
            time.sleep(1 / 12)

    # ── Главный поток: отрисовка ──────────────────────────────────────────
    def _refresh_settings(self):
        """
        Переложить значения полей в обычный словарь.

        Вызывается только из главного потока — это единственное место, где
        читаются tkinter-переменные оверлея.
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
        # Делить на длину окна (2 с) нельзя: пока окно не заполнилось, счётчик
        # занижает — на первой секунде показывал половину настоящего FPS.
        # Считаем по фактически прошедшему времени: N-1 интервалов между
        # N кадрами.
        span = self._fps_times[-1] - self._fps_times[0]
        fps = (len(self._fps_times) - 1) / span if span > 1e-9 else 0.0

        text = (f"окон: {payload['n_tiles']}\n"
                f"инференс: {payload['infer_ms']:.0f} мс\n"
                f"FPS: {fps:.1f}\n"
                f"кадр: {frame.shape[1]}x{frame.shape[0]}")
        if self._running_model:
            branch, engine, _ = self._running_model
            text += f"\nмодель: {branch}/{engine}"
            if self.pick_model.get() != ACTIVE_CHOICE:
                text += f"\n{self.pick_model.get().strip()[:22]}"
        if payload['accuracy'] is not None:
            text += f"\n\nточность разметки:\n{payload['accuracy'] * 100:.1f}%"
        if self.rec_var.get() or self.recording:
            text += (f"\n\nзаписано: {len(self.recording)} кадров"
                     f"\n({self.recording_bytes / 1024 / 1024:.0f} МБ)")
            if self.record_full:
                text += "\nБУФЕР ПОЛОН — запись\nостановлена"
        self.info.config(text=text)

    def _record(self, frame: np.ndarray):
        """Дописать кадр в буфер записи, если тот ещё не переполнен."""
        if self.record_full:
            return
        if (len(self.recording) >= self.MAX_RECORD_FRAMES
                or self.recording_bytes + frame.nbytes > self.MAX_RECORD_BYTES):
            self.record_full = True
            messagebox.showwarning(
                "Буфер записи полон",
                f"Записано {len(self.recording)} кадров "
                f"({self.recording_bytes / 1024 / 1024:.0f} МБ) — предел.\n\n"
                f"Кадры копятся в памяти, поэтому у буфера есть потолок "
                f"({self.MAX_RECORD_FRAMES} кадров или "
                f"{self.MAX_RECORD_BYTES // 1024 // 1024} МБ).\n"
                f"Сохраните видео и снимите галку «писать результат», "
                f"чтобы начать новую запись.")
            return
        self.recording.append(frame)
        self.recording_bytes += frame.nbytes

    # ── Сохранение ────────────────────────────────────────────────────────
    def save_frame(self):
        """
        Сохранить размеченный кадр КАК ЕСТЬ, пиксель в пиксель.

        Раньше кадр прогонялся через matplotlib-фигуру размером dpi=100, а
        сохранялся с dpi=150: на диск ложилась картинка в полтора раза крупнее
        исходной и пересэмплированная — то есть не те пиксели, которые видел
        и оценивал пользователь. Для отчёта, где по этому кадру считают
        разметку, это недопустимо.
        """
        if self.last_frame is None:
            messagebox.showinfo("Нет кадра", "Сначала запустите источник.")
            return
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR,
                            f"overlay_{time.strftime('%Y%m%d_%H%M%S')}.png")
        try:
            written = vision.write_image(path, self.last_frame)
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("Не удалось сохранить кадр", str(exc))
            return
        h, w = self.last_frame.shape[:2]
        self.info.config(text=f"кадр {w}x{h} -> {written}")

    def save_video(self):
        if not self.recording:
            messagebox.showinfo(
                "Нечего сохранять",
                "Включите «писать результат» и запустите источник.")
            return
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = os.path.join(config.RESULTS_DIR,
                            f"overlay_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        try:
            vision.write_video(path, self.recording)
        except vision.VideoUnavailable as exc:
            messagebox.showerror("Нет OpenCV", str(exc))
            return
        self.info.config(text=f"видео ({len(self.recording)} кадров)\n-> {path}")

    def on_event(self, event):
        if event['type'] == 'error':
            messagebox.showerror("Ошибка видео", event['text'])
            self.info.config(text="ошибка")
            self.btn_stop.config(state=tk.DISABLED)
        elif event['type'] == 'done':
            self.btn_stop.config(state=tk.DISABLED)


# ── Вкладка: Тест (контейнер) ─────────────────────────────────────────────────

class TestTab:
    """Контейнер с двумя режимами теста — фото и видео."""

    def __init__(self, root: tk.Tk, parent: tk.Frame, lock: BusyLock | None = None):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)
        photo = tk.Frame(nb, bg=BG)
        video = tk.Frame(nb, bg=BG)
        nb.add(photo, text="  Фото  ")
        nb.add(video, text="  Видео  ")
        self.photo = PhotoTab(root, photo, lock=lock)
        self.video = VideoTab(root, video, lock=lock)


# ── Оформление Notebook ───────────────────────────────────────────────────────

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
    # Таблица библиотеки моделей: тот же тёмный фон, что и у остального окна.
    style.configure("Treeview", background=BG2, fieldbackground=BG2,
                    foreground=FG, borderwidth=0, font=("Consolas", 8),
                    rowheight=20)
    style.configure("Treeview.Heading", background=BG3, foreground=ACCENT,
                    font=("Consolas", 8, "bold"), borderwidth=0)
    style.map("Treeview", background=[("selected", BG3)],
              foreground=[("selected", ACCENT)])


# ── Точка входа ───────────────────────────────────────────────────────────────

def main(initial_tab: str | None = None) -> None:
    # Датасет применяется ДО создания вкладок: от него зависят имена классов,
    # их число и все подписи, которые вкладки строят в конструкторе.
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
    for key, text in [("data", "  Данные  "),
                      ("features", "  Признаки  "),
                      ("train", "  Обучение  "),
                      ("distill", "  Дистилляция  "),
                      ("models", "  Модели  "),
                      ("test", "  Тест  ")]:
        frames[key] = tk.Frame(nb, bg=BG)
        nb.add(frames[key], text=text)

    # Одна блокировка на всё приложение: пока одна вкладка занята тяжёлой
    # задачей, остальные не дают запустить свою.
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

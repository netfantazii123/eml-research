"""
Flappy Bird AI — графический интерфейс на PySide6.

Три вкладки:
    1. Тренировка   — гиперпараметры, запуск train/test, live-лог, прогресс.
    2. Демо         — выбор сохранённых моделей/формул, запуск pygame-демо.
    3. Библиотека   — список всех сохранённых артефактов с деталями.
"""

import os
import sys
import threading
import traceback
from datetime import datetime

from PySide6.QtCore import Qt, QObject, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QAction, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPlainTextEdit, QTreeWidget, QTreeWidgetItem,
    QListWidget, QListWidgetItem, QSplitter, QGroupBox,
    QMessageBox, QFileDialog, QProgressBar, QStatusBar,
)

import config
import storage
from storage import (
    list_saved_runs, load_eml_formula, load_ga_model,
    latest_ga_path,
)


# ── Captura stdout → Qt signal ──────────────────────────────────────────────

class _StreamToSignal:
    """Подменяет sys.stdout: каждая строка эмитится через Qt-сигнал."""

    def __init__(self, signal: Signal):
        self.signal = signal
        self._buf = ''

    def write(self, s: str):
        if not s:
            return
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            try:
                self.signal.emit(line)
            except Exception:
                pass

    def flush(self):
        if self._buf:
            try:
                self.signal.emit(self._buf)
            except Exception:
                pass
            self._buf = ''


# ── Worker (тренировка в отдельном потоке) ──────────────────────────────────

class _WorkerSignals(QObject):
    log = Signal(str)
    finished = Signal(object)   # results dict | None
    error = Signal(str)
    progress = Signal(str)      # текущий шаг (строка для статус-бара)


class TrainingWorker:
    """
    Запускает run_benchmark или smoke-тест в фоновом потоке.

    Лог идёт через перехват stdout → signal.log.
    """

    def __init__(self, mode: str, overrides: dict | None = None):
        self.mode = mode  # 'benchmark' | 'test'
        self.overrides = overrides or {}
        self.signals = _WorkerSignals()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _apply_overrides(self):
        for k, v in self.overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)

    def _run(self):
        try:
            self._apply_overrides()
            old_stdout = sys.stdout
            sys.stdout = _StreamToSignal(self.signals.log)
            try:
                if self.overrides:
                    print("Применённые параметры:")
                    for k, v in self.overrides.items():
                        print(f"  config.{k} = {v}")
                if self.mode == 'benchmark':
                    self.signals.progress.emit("Бенчмарк запущен...")
                    from benchmark import run_benchmark
                    results = run_benchmark(verbose=True)
                    self.signals.finished.emit(results)
                elif self.mode == 'test':
                    self.signals.progress.emit("Smoke-тест запущен...")
                    from main import cmd_test
                    cmd_test()
                    self.signals.finished.emit(None)
                else:
                    self.signals.error.emit(f"Unknown mode: {self.mode}")
            finally:
                if hasattr(sys.stdout, 'flush'):
                    sys.stdout.flush()
                sys.stdout = old_stdout
        except Exception:
            self.signals.error.emit(traceback.format_exc())


# ── Вкладка: Тренировка ─────────────────────────────────────────────────────

class TrainTab(QWidget):
    """Гиперпараметры + кнопки запуска + live-лог."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: TrainingWorker | None = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)

        # ── Левая колонка: гиперпараметры ────────────────────────────────
        left = QVBoxLayout()

        ga_box = QGroupBox("Генетический алгоритм (GA)")
        ga_form = QFormLayout(ga_box)
        self.ga_pop = QSpinBox(); self.ga_pop.setRange(10, 100000); self.ga_pop.setValue(config.GA_POPULATION)
        self.ga_gen = QSpinBox(); self.ga_gen.setRange(1, 100000); self.ga_gen.setValue(config.GA_GENERATIONS)
        self.ga_mut_rate = QDoubleSpinBox(); self.ga_mut_rate.setRange(0.0, 1.0); self.ga_mut_rate.setSingleStep(0.01); self.ga_mut_rate.setDecimals(3); self.ga_mut_rate.setValue(config.GA_MUTATION_RATE)
        self.ga_mut_std = QDoubleSpinBox(); self.ga_mut_std.setRange(0.0, 5.0); self.ga_mut_std.setSingleStep(0.05); self.ga_mut_std.setDecimals(3); self.ga_mut_std.setValue(config.GA_MUTATION_STD)
        self.ga_elite = QSpinBox(); self.ga_elite.setRange(0, 100); self.ga_elite.setValue(config.GA_ELITISM)
        self.ga_max_frames = QSpinBox(); self.ga_max_frames.setRange(100, 1000000); self.ga_max_frames.setValue(config.GA_MAX_FRAMES)
        self.ga_patience = QSpinBox(); self.ga_patience.setRange(0, 100000); self.ga_patience.setValue(getattr(config, 'GA_PATIENCE', 15))
        self.ga_target = QSpinBox(); self.ga_target.setRange(0, 100000); self.ga_target.setValue(getattr(config, 'GA_TARGET_SCORE', 0))
        ga_form.addRow("Популяция", self.ga_pop)
        ga_form.addRow("Поколений (max)", self.ga_gen)
        ga_form.addRow("Mutation rate", self.ga_mut_rate)
        ga_form.addRow("Mutation std", self.ga_mut_std)
        ga_form.addRow("Элитизм", self.ga_elite)
        ga_form.addRow("Max frames/game", self.ga_max_frames)
        ga_form.addRow("Patience (autostop)", self.ga_patience)
        ga_form.addRow("Target score (0=off)", self.ga_target)
        left.addWidget(ga_box)

        eml_box = QGroupBox("EML дистилляция")
        eml_form = QFormLayout(eml_box)
        self.eml_pop = QSpinBox(); self.eml_pop.setRange(10, 100000); self.eml_pop.setValue(config.EML_POPULATION)
        self.eml_gen = QSpinBox(); self.eml_gen.setRange(1, 100000); self.eml_gen.setValue(config.EML_GENERATIONS)
        self.eml_elite = QSpinBox(); self.eml_elite.setRange(0, 100); self.eml_elite.setValue(config.EML_ELITISM)
        self.eml_depth = QSpinBox(); self.eml_depth.setRange(1, 15); self.eml_depth.setValue(config.EML_MAX_DEPTH)
        self.eml_patience = QSpinBox(); self.eml_patience.setRange(1, 100000); self.eml_patience.setValue(config.EML_PATIENCE)
        self.eml_dataset = QSpinBox(); self.eml_dataset.setRange(1, 100000); self.eml_dataset.setValue(config.EML_DATASET_EPISODES)
        eml_form.addRow("Популяция", self.eml_pop)
        eml_form.addRow("Поколений", self.eml_gen)
        eml_form.addRow("Элитизм", self.eml_elite)
        eml_form.addRow("Макс. глубина", self.eml_depth)
        eml_form.addRow("Patience", self.eml_patience)
        eml_form.addRow("Episodes для датасета", self.eml_dataset)
        left.addWidget(eml_box)

        # Кнопки запуска
        btn_box = QGroupBox("Запуск")
        btn_v = QVBoxLayout(btn_box)
        self.btn_train = QPushButton("▶  Полный бенчмарк (GA + 3 EML)")
        self.btn_train.setMinimumHeight(36)
        self.btn_train.clicked.connect(self._on_train_clicked)
        self.btn_test = QPushButton("▶  Smoke-тест (быстрая проверка)")
        self.btn_test.clicked.connect(self._on_test_clicked)
        self.btn_reset = QPushButton("⟲  Сбросить параметры к дефолтным")
        self.btn_reset.clicked.connect(self._reset_defaults)
        btn_v.addWidget(self.btn_train)
        btn_v.addWidget(self.btn_test)
        btn_v.addWidget(self.btn_reset)
        left.addWidget(btn_box)

        left.addStretch(1)

        # ── Правая колонка: лог ──────────────────────────────────────────
        right = QVBoxLayout()
        log_label = QLabel("Live лог тренировки")
        log_label.setStyleSheet("font-weight: bold;")
        right.addWidget(log_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        mono = QFont("Consolas", 9)
        if not mono.exactMatch():
            mono = QFont("Courier New", 9)
        self.log_view.setFont(mono)
        self.log_view.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        right.addWidget(self.log_view, 1)

        log_btn_row = QHBoxLayout()
        self.btn_clear_log = QPushButton("Очистить лог")
        self.btn_clear_log.clicked.connect(self.log_view.clear)
        self.btn_save_log = QPushButton("Сохранить лог в файл...")
        self.btn_save_log.clicked.connect(self._save_log)
        log_btn_row.addWidget(self.btn_clear_log)
        log_btn_row.addWidget(self.btn_save_log)
        log_btn_row.addStretch(1)
        right.addLayout(log_btn_row)

        # Splitter
        left_w = QWidget(); left_w.setLayout(left)
        right_w = QWidget(); right_w.setLayout(right)
        splitter = QSplitter()
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 700])
        root.addWidget(splitter)

    # ── Действия ─────────────────────────────────────────────────────────

    def _collect_overrides(self) -> dict:
        return {
            'GA_POPULATION':       self.ga_pop.value(),
            'GA_GENERATIONS':      self.ga_gen.value(),
            'GA_MUTATION_RATE':    self.ga_mut_rate.value(),
            'GA_MUTATION_STD':     self.ga_mut_std.value(),
            'GA_ELITISM':          self.ga_elite.value(),
            'GA_MAX_FRAMES':       self.ga_max_frames.value(),
            'GA_PATIENCE':         self.ga_patience.value(),
            'GA_TARGET_SCORE':     self.ga_target.value(),
            'EML_POPULATION':      self.eml_pop.value(),
            'EML_GENERATIONS':     self.eml_gen.value(),
            'EML_ELITISM':         self.eml_elite.value(),
            'EML_MAX_DEPTH':       self.eml_depth.value(),
            'EML_PATIENCE':        self.eml_patience.value(),
            'EML_DATASET_EPISODES': self.eml_dataset.value(),
        }

    def _reset_defaults(self):
        import importlib
        importlib.reload(config)
        self.ga_pop.setValue(config.GA_POPULATION)
        self.ga_gen.setValue(config.GA_GENERATIONS)
        self.ga_mut_rate.setValue(config.GA_MUTATION_RATE)
        self.ga_mut_std.setValue(config.GA_MUTATION_STD)
        self.ga_elite.setValue(config.GA_ELITISM)
        self.ga_max_frames.setValue(config.GA_MAX_FRAMES)
        self.ga_patience.setValue(getattr(config, 'GA_PATIENCE', 15))
        self.ga_target.setValue(getattr(config, 'GA_TARGET_SCORE', 0))
        self.eml_pop.setValue(config.EML_POPULATION)
        self.eml_gen.setValue(config.EML_GENERATIONS)
        self.eml_elite.setValue(config.EML_ELITISM)
        self.eml_depth.setValue(config.EML_MAX_DEPTH)
        self.eml_patience.setValue(config.EML_PATIENCE)
        self.eml_dataset.setValue(config.EML_DATASET_EPISODES)

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_train.setEnabled(enabled)
        self.btn_test.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)

    def _on_train_clicked(self):
        self._start_worker('benchmark', self._collect_overrides())

    def _on_test_clicked(self):
        self._start_worker('test', {})

    def _start_worker(self, mode: str, overrides: dict):
        if self._worker is not None and self._worker._thread and self._worker._thread.is_alive():
            QMessageBox.warning(self, "Занято", "Уже идёт тренировка. Дождитесь окончания.")
            return
        self._set_buttons_enabled(False)
        self.log_view.appendPlainText(f"\n>>> Запуск: {mode}  ({datetime.now():%H:%M:%S})\n")
        self._worker = TrainingWorker(mode, overrides)
        self._worker.signals.log.connect(self._on_log)
        self._worker.signals.finished.connect(self._on_finished)
        self._worker.signals.error.connect(self._on_error)
        main = self.window()
        if isinstance(main, MainWindow):
            self._worker.signals.progress.connect(main.set_status)
        self._worker.start()

    @Slot(str)
    def _on_log(self, line: str):
        self.log_view.appendPlainText(line)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot(object)
    def _on_finished(self, results):
        self._set_buttons_enabled(True)
        self.log_view.appendPlainText("\n<<< Завершено\n")
        main = self.window()
        if isinstance(main, MainWindow):
            main.set_status("Готово")
            main.library_tab.refresh()

    @Slot(str)
    def _on_error(self, tb_text: str):
        self._set_buttons_enabled(True)
        self.log_view.appendPlainText("\n!!! ОШИБКА:\n" + tb_text + "\n")
        main = self.window()
        if isinstance(main, MainWindow):
            main.set_status("Ошибка")

    def _save_log(self):
        text = self.log_view.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "Лог пуст", "Нечего сохранять.")
            return
        ts = storage.make_timestamp()
        default = os.path.join(storage.LOGS_DIR, f"gui_log_{ts}.log")
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить лог", default, "Log (*.log *.txt)")
        if not path:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        QMessageBox.information(self, "Сохранено", f"Лог сохранён:\n{path}")


# ── Вкладка: Демо ───────────────────────────────────────────────────────────

class DemoTab(QWidget):
    """Выбор моделей и запуск pygame-демо."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)

        info = QLabel(
            "Выберите модели/формулы для одновременной демонстрации. "
            "Pygame откроется в отдельном окне."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        root.addWidget(self.list_widget, 1)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Обновить список")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_select_latest = QPushButton("Выбрать последнее каждого метода")
        self.btn_select_latest.clicked.connect(self._select_latest_per_method)
        self.btn_run = QPushButton("▶  Запустить демо")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self._run_demo)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_select_latest)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_run)
        root.addLayout(btn_row)

        hint = QLabel("В окне демо: R = рестарт, SPACE = пауза, ESC = выйти.")
        hint.setStyleSheet("color: gray;")
        root.addWidget(hint)

    def refresh(self):
        self.list_widget.clear()
        runs = list_saved_runs()
        if not runs:
            item = QListWidgetItem("(пусто) — сначала запустите тренировку")
            item.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(item)
            return
        for r in runs:
            score_str = f"score={r['score']}" if r['score'] is not None else "no-score"
            label = f"[{r['method']:>10}]  {r['date']} {r['time']}  {score_str}"
            if r['kind'] == 'eml' and r['extra'].get('formula'):
                f = r['extra']['formula']
                if len(f) > 60:
                    f = f[:57] + '...'
                label += f"   {f}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, r)
            self.list_widget.addItem(item)

    def _select_latest_per_method(self):
        self.list_widget.clearSelection()
        seen = set()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            r = item.data(Qt.UserRole)
            if not r:
                continue
            method = r['method']
            if method in seen:
                continue
            seen.add(method)
            item.setSelected(True)

    def _run_demo(self):
        selected = [self.list_widget.item(i).data(Qt.UserRole)
                    for i in range(self.list_widget.count())
                    if self.list_widget.item(i).isSelected()
                    and self.list_widget.item(i).data(Qt.UserRole)]
        if not selected:
            QMessageBox.warning(self, "Не выбрано", "Выберите хотя бы одну модель.")
            return

        # Загрузка агентов
        from demo import load_agents_from_paths
        paths = [r['path'] for r in selected]
        try:
            agents = load_agents_from_paths(paths)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))
            return
        if not agents:
            QMessageBox.warning(self, "Пусто", "Не удалось загрузить ни одного агента.")
            return

        # pygame нужно запускать в главном потоке. Это блокирует GUI пока окно открыто.
        # Делаем красивый дисклеймер.
        main = self.window()
        if isinstance(main, MainWindow):
            main.set_status("Демо запущено (окно pygame)...")
        try:
            from demo import run_demo as _run
            _run(agents=agents)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка демо", traceback.format_exc())
        finally:
            if isinstance(main, MainWindow):
                main.set_status("Готово")


# ── Вкладка: Библиотека ─────────────────────────────────────────────────────

class LibraryTab(QWidget):
    """Полный список сохранённого + детали + удаление."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("Все методы", None)
        self.filter_combo.addItem("Только GA", "GA")
        self.filter_combo.addItem("EML-weak", "eml-weak")
        self.filter_combo.addItem("EML-medium", "eml-medium")
        self.filter_combo.addItem("EML-strong", "eml-strong")
        self.filter_combo.currentIndexChanged.connect(self.refresh)
        self.btn_refresh = QPushButton("Обновить")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_logs_dir = QPushButton("Открыть logs/")
        self.btn_logs_dir.clicked.connect(lambda: self._open_dir(storage.LOGS_DIR))
        self.btn_models_dir = QPushButton("Открыть models/")
        self.btn_models_dir.clicked.connect(lambda: self._open_dir(config.MODELS_DIR))
        self.btn_results_dir = QPushButton("Открыть results/")
        self.btn_results_dir.clicked.connect(lambda: self._open_dir(config.RESULTS_DIR))
        top_row.addWidget(QLabel("Фильтр:"))
        top_row.addWidget(self.filter_combo)
        top_row.addWidget(self.btn_refresh)
        top_row.addStretch(1)
        top_row.addWidget(self.btn_models_dir)
        top_row.addWidget(self.btn_results_dir)
        top_row.addWidget(self.btn_logs_dir)
        root.addLayout(top_row)

        splitter = QSplitter()

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Метод", "Дата", "Время", "Score", "Файл"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(True)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.tree.setColumnWidth(0, 110)
        self.tree.setColumnWidth(1, 100)
        self.tree.setColumnWidth(2, 80)
        self.tree.setColumnWidth(3, 60)
        splitter.addWidget(self.tree)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        mono = QFont("Consolas", 9)
        if not mono.exactMatch():
            mono = QFont("Courier New", 9)
        self.details.setFont(mono)
        splitter.addWidget(self.details)
        splitter.setSizes([600, 380])

        root.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self.btn_delete = QPushButton("🗑  Удалить выбранное")
        self.btn_delete.clicked.connect(self._delete_selected)
        bottom.addWidget(self.btn_delete)
        bottom.addStretch(1)
        root.addLayout(bottom)

    def refresh(self):
        self.tree.clear()
        method_filter = self.filter_combo.currentData()
        for r in list_saved_runs():
            if method_filter and r['method'] != method_filter:
                continue
            item = QTreeWidgetItem([
                r['method'],
                r['date'],
                r['time'],
                str(r['score']) if r['score'] is not None else '—',
                os.path.basename(r['path']),
            ])
            item.setData(0, Qt.UserRole, r)
            self.tree.addTopLevelItem(item)
        self.details.clear()

    def _on_select(self):
        items = self.tree.selectedItems()
        if not items:
            self.details.clear()
            return
        r = items[0].data(0, Qt.UserRole)
        if not r:
            return
        lines = [
            f"Метод:     {r['method']}",
            f"Дата:      {r['date']} {r['time']}",
            f"Score:     {r['score']}",
            f"Файл:      {r['path']}",
            "",
        ]
        ex = r.get('extra', {}) or {}
        if r['kind'] == 'eml':
            lines += [
                f"Mode:          {ex.get('mode')}",
                f"Depth penalty: {ex.get('depth_penalty')}",
                f"Depth:         {ex.get('depth')}",
                f"Size:          {ex.get('size')}",
                f"N vars:        {ex.get('n_vars')}",
                f"Elapsed:       {self._fmt_elapsed(ex.get('elapsed'))}",
                f"Test scores:   {ex.get('test_scores')}",
                "",
                "ФОРМУЛА:",
                ex.get('formula', '') or '(нет)',
            ]
        else:
            lines += [
                f"Arch:        {ex.get('arch', '4-16-1')}",
                f"Params:      {ex.get('params', '—')}",
                f"Elapsed:     {self._fmt_elapsed(ex.get('elapsed'))}",
                f"Total fr.:   {ex.get('total_frames', '—')}",
                f"Test scores: {ex.get('test_scores', '—')}",
            ]
        self.details.setPlainText('\n'.join(lines))

    @staticmethod
    def _fmt_elapsed(v):
        if v is None:
            return '—'
        try:
            v = float(v)
        except (TypeError, ValueError):
            return str(v)
        if v < 60:
            return f"{v:.1f} s"
        return f"{int(v // 60)}m {int(v % 60)}s"

    def _delete_selected(self):
        items = self.tree.selectedItems()
        if not items:
            return
        names = [it.text(4) for it in items]
        msg = "Удалить файлы?\n\n" + "\n".join(names)
        if QMessageBox.question(self, "Подтверждение", msg) != QMessageBox.Yes:
            return
        for it in items:
            r = it.data(0, Qt.UserRole)
            if not r:
                continue
            path = r['path']
            try:
                os.remove(path)
                # Удалить sidecar .meta.json у .pt
                if path.endswith('.pt'):
                    meta = path[:-3] + '.meta.json'
                    if os.path.exists(meta):
                        os.remove(meta)
            except OSError as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось удалить {path}:\n{e}")
        self.refresh()
        main = self.window()
        if isinstance(main, MainWindow):
            main.demo_tab.refresh()

    @staticmethod
    def _open_dir(path: str):
        os.makedirs(path, exist_ok=True)
        if sys.platform.startswith('win'):
            os.startfile(os.path.abspath(path))  # noqa: S606
        elif sys.platform == 'darwin':
            os.system(f'open "{os.path.abspath(path)}"')  # noqa: S605
        else:
            os.system(f'xdg-open "{os.path.abspath(path)}"')  # noqa: S605


# ── Главное окно ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flappy Bird AI — Control Panel")
        self.resize(1180, 720)

        tabs = QTabWidget()
        self.train_tab = TrainTab()
        self.demo_tab = DemoTab()
        self.library_tab = LibraryTab()
        tabs.addTab(self.train_tab, "Тренировка")
        tabs.addTab(self.demo_tab, "Демо")
        tabs.addTab(self.library_tab, "Библиотека")
        self.setCentralWidget(tabs)

        self.setStatusBar(QStatusBar())
        self.set_status("Готово")

    def set_status(self, text: str):
        self.statusBar().showMessage(text)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

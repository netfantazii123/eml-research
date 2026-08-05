"""
Flappy Bird AI — a PySide6 graphical interface.

Three tabs:
    1. Training  — hyperparameters, train/test launch, live log, progress.
    2. Demo      — selection of saved models/formulas, pygame demo launch.
    3. Library   — a list of all saved artifacts with details.
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
    """Replaces sys.stdout: every line is emitted through a Qt signal."""

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


# ── Worker (training in a separate thread) ──────────────────────────────────

class _WorkerSignals(QObject):
    log = Signal(str)
    finished = Signal(object)   # results dict | None
    error = Signal(str)
    progress = Signal(str)      # current step (a line for the status bar)


class TrainingWorker:
    """
    Runs run_benchmark or the smoke test in a background thread.

    The log goes through an stdout intercept -> signal.log.
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
                    print("Applied parameters:")
                    for k, v in self.overrides.items():
                        print(f"  config.{k} = {v}")
                if self.mode == 'benchmark':
                    self.signals.progress.emit("Benchmark started...")
                    from benchmark import run_benchmark
                    results = run_benchmark(verbose=True)
                    self.signals.finished.emit(results)
                elif self.mode == 'test':
                    self.signals.progress.emit("Smoke test started...")
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


# ── Tab: Training ───────────────────────────────────────────────────────────

class TrainTab(QWidget):
    """Hyperparameters + launch buttons + live log."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: TrainingWorker | None = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)

        # ── Left column: hyperparameters ─────────────────────────────────
        left = QVBoxLayout()

        ga_box = QGroupBox("Genetic algorithm (GA)")
        ga_form = QFormLayout(ga_box)
        self.ga_pop = QSpinBox(); self.ga_pop.setRange(10, 100000); self.ga_pop.setValue(config.GA_POPULATION)
        self.ga_gen = QSpinBox(); self.ga_gen.setRange(1, 100000); self.ga_gen.setValue(config.GA_GENERATIONS)
        self.ga_mut_rate = QDoubleSpinBox(); self.ga_mut_rate.setRange(0.0, 1.0); self.ga_mut_rate.setSingleStep(0.01); self.ga_mut_rate.setDecimals(3); self.ga_mut_rate.setValue(config.GA_MUTATION_RATE)
        self.ga_mut_std = QDoubleSpinBox(); self.ga_mut_std.setRange(0.0, 5.0); self.ga_mut_std.setSingleStep(0.05); self.ga_mut_std.setDecimals(3); self.ga_mut_std.setValue(config.GA_MUTATION_STD)
        self.ga_elite = QSpinBox(); self.ga_elite.setRange(0, 100); self.ga_elite.setValue(config.GA_ELITISM)
        self.ga_max_frames = QSpinBox(); self.ga_max_frames.setRange(100, 1000000); self.ga_max_frames.setValue(config.GA_MAX_FRAMES)
        self.ga_patience = QSpinBox(); self.ga_patience.setRange(0, 100000); self.ga_patience.setValue(getattr(config, 'GA_PATIENCE', 15))
        self.ga_target = QSpinBox(); self.ga_target.setRange(0, 100000); self.ga_target.setValue(getattr(config, 'GA_TARGET_SCORE', 0))
        ga_form.addRow("Population", self.ga_pop)
        ga_form.addRow("Generations (max)", self.ga_gen)
        ga_form.addRow("Mutation rate", self.ga_mut_rate)
        ga_form.addRow("Mutation std", self.ga_mut_std)
        ga_form.addRow("Elitism", self.ga_elite)
        ga_form.addRow("Max frames/game", self.ga_max_frames)
        ga_form.addRow("Patience (autostop)", self.ga_patience)
        ga_form.addRow("Target score (0=off)", self.ga_target)
        left.addWidget(ga_box)

        eml_box = QGroupBox("EML distillation")
        eml_form = QFormLayout(eml_box)
        self.eml_pop = QSpinBox(); self.eml_pop.setRange(10, 100000); self.eml_pop.setValue(config.EML_POPULATION)
        self.eml_gen = QSpinBox(); self.eml_gen.setRange(1, 100000); self.eml_gen.setValue(config.EML_GENERATIONS)
        self.eml_elite = QSpinBox(); self.eml_elite.setRange(0, 100); self.eml_elite.setValue(config.EML_ELITISM)
        self.eml_depth = QSpinBox(); self.eml_depth.setRange(1, 15); self.eml_depth.setValue(config.EML_MAX_DEPTH)
        self.eml_patience = QSpinBox(); self.eml_patience.setRange(1, 100000); self.eml_patience.setValue(config.EML_PATIENCE)
        self.eml_dataset = QSpinBox(); self.eml_dataset.setRange(1, 100000); self.eml_dataset.setValue(config.EML_DATASET_EPISODES)
        eml_form.addRow("Population", self.eml_pop)
        eml_form.addRow("Generations", self.eml_gen)
        eml_form.addRow("Elitism", self.eml_elite)
        eml_form.addRow("Max depth", self.eml_depth)
        eml_form.addRow("Patience", self.eml_patience)
        eml_form.addRow("Episodes for the dataset", self.eml_dataset)
        left.addWidget(eml_box)

        # Launch buttons
        btn_box = QGroupBox("Launch")
        btn_v = QVBoxLayout(btn_box)
        self.btn_train = QPushButton("▶  Full benchmark (GA + 3 EML)")
        self.btn_train.setMinimumHeight(36)
        self.btn_train.clicked.connect(self._on_train_clicked)
        self.btn_test = QPushButton("▶  Smoke test (a quick check)")
        self.btn_test.clicked.connect(self._on_test_clicked)
        self.btn_reset = QPushButton("⟲  Reset parameters to defaults")
        self.btn_reset.clicked.connect(self._reset_defaults)
        btn_v.addWidget(self.btn_train)
        btn_v.addWidget(self.btn_test)
        btn_v.addWidget(self.btn_reset)
        left.addWidget(btn_box)

        left.addStretch(1)

        # ── Right column: the log ────────────────────────────────────────
        right = QVBoxLayout()
        log_label = QLabel("Live training log")
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
        self.btn_clear_log = QPushButton("Clear log")
        self.btn_clear_log.clicked.connect(self.log_view.clear)
        self.btn_save_log = QPushButton("Save log to a file...")
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

    # ── Actions ──────────────────────────────────────────────────────────

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
            QMessageBox.warning(self, "Busy", "Training is already running. Wait for it to finish.")
            return
        self._set_buttons_enabled(False)
        self.log_view.appendPlainText(f"\n>>> Start: {mode}  ({datetime.now():%H:%M:%S})\n")
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
        self.log_view.appendPlainText("\n<<< Finished\n")
        main = self.window()
        if isinstance(main, MainWindow):
            main.set_status("Ready")
            main.library_tab.refresh()

    @Slot(str)
    def _on_error(self, tb_text: str):
        self._set_buttons_enabled(True)
        self.log_view.appendPlainText("\n!!! ERROR:\n" + tb_text + "\n")
        main = self.window()
        if isinstance(main, MainWindow):
            main.set_status("Error")

    def _save_log(self):
        text = self.log_view.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "Log is empty", "Nothing to save.")
            return
        ts = storage.make_timestamp()
        default = os.path.join(storage.LOGS_DIR, f"gui_log_{ts}.log")
        path, _ = QFileDialog.getSaveFileName(self, "Save log", default, "Log (*.log *.txt)")
        if not path:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        QMessageBox.information(self, "Saved", f"Log saved:\n{path}")


# ── Tab: Demo ───────────────────────────────────────────────────────────────

class DemoTab(QWidget):
    """Model selection and pygame demo launch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)

        info = QLabel(
            "Select models/formulas to demonstrate simultaneously. "
            "Pygame will open in a separate window."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        root.addWidget(self.list_widget, 1)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh list")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_select_latest = QPushButton("Select the latest of each method")
        self.btn_select_latest.clicked.connect(self._select_latest_per_method)
        self.btn_run = QPushButton("▶  Run demo")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self._run_demo)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_select_latest)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_run)
        root.addLayout(btn_row)

        hint = QLabel("In the demo window: R = restart, SPACE = pause, ESC = quit.")
        hint.setStyleSheet("color: gray;")
        root.addWidget(hint)

    def refresh(self):
        self.list_widget.clear()
        runs = list_saved_runs()
        if not runs:
            item = QListWidgetItem("(empty) — run the training first")
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
            QMessageBox.warning(self, "Nothing selected", "Select at least one model.")
            return

        # Loading the agents
        from demo import load_agents_from_paths
        paths = [r['path'] for r in selected]
        try:
            agents = load_agents_from_paths(paths)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return
        if not agents:
            QMessageBox.warning(self, "Empty", "Could not load a single agent.")
            return

        # pygame must run in the main thread. This blocks the GUI while the
        # window is open, so the status bar says so explicitly.
        main = self.window()
        if isinstance(main, MainWindow):
            main.set_status("Demo running (pygame window)...")
        try:
            from demo import run_demo as _run
            _run(agents=agents)
        except Exception as e:
            QMessageBox.critical(self, "Demo error", traceback.format_exc())
        finally:
            if isinstance(main, MainWindow):
                main.set_status("Ready")


# ── Tab: Library ────────────────────────────────────────────────────────────

class LibraryTab(QWidget):
    """Full list of saved artifacts + details + deletion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All methods", None)
        self.filter_combo.addItem("GA only", "GA")
        self.filter_combo.addItem("EML-weak", "eml-weak")
        self.filter_combo.addItem("EML-medium", "eml-medium")
        self.filter_combo.addItem("EML-strong", "eml-strong")
        self.filter_combo.currentIndexChanged.connect(self.refresh)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_logs_dir = QPushButton("Open logs/")
        self.btn_logs_dir.clicked.connect(lambda: self._open_dir(storage.LOGS_DIR))
        self.btn_models_dir = QPushButton("Open models/")
        self.btn_models_dir.clicked.connect(lambda: self._open_dir(config.MODELS_DIR))
        self.btn_results_dir = QPushButton("Open results/")
        self.btn_results_dir.clicked.connect(lambda: self._open_dir(config.RESULTS_DIR))
        top_row.addWidget(QLabel("Filter:"))
        top_row.addWidget(self.filter_combo)
        top_row.addWidget(self.btn_refresh)
        top_row.addStretch(1)
        top_row.addWidget(self.btn_models_dir)
        top_row.addWidget(self.btn_results_dir)
        top_row.addWidget(self.btn_logs_dir)
        root.addLayout(top_row)

        splitter = QSplitter()

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Method", "Date", "Time", "Score", "File"])
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
        self.btn_delete = QPushButton("🗑  Delete selected")
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
            f"Method:    {r['method']}",
            f"Date:      {r['date']} {r['time']}",
            f"Score:     {r['score']}",
            f"File:      {r['path']}",
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
                "FORMULA:",
                ex.get('formula', '') or '(none)',
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
        msg = "Delete the files?\n\n" + "\n".join(names)
        if QMessageBox.question(self, "Confirmation", msg) != QMessageBox.Yes:
            return
        for it in items:
            r = it.data(0, Qt.UserRole)
            if not r:
                continue
            path = r['path']
            try:
                os.remove(path)
                # Delete the sidecar .meta.json of a .pt
                if path.endswith('.pt'):
                    meta = path[:-3] + '.meta.json'
                    if os.path.exists(meta):
                        os.remove(meta)
            except OSError as e:
                QMessageBox.warning(self, "Error", f"Could not delete {path}:\n{e}")
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


# ── Main window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flappy Bird AI — Control Panel")
        self.resize(1180, 720)

        tabs = QTabWidget()
        self.train_tab = TrainTab()
        self.demo_tab = DemoTab()
        self.library_tab = LibraryTab()
        tabs.addTab(self.train_tab, "Training")
        tabs.addTab(self.demo_tab, "Demo")
        tabs.addTab(self.library_tab, "Library")
        self.setCentralWidget(tabs)

        self.setStatusBar(QStatusBar())
        self.set_status("Ready")

    def set_status(self, text: str):
        self.statusBar().showMessage(text)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

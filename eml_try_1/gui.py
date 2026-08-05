"""
EML Calculator — a GUI application demonstrating the EML transformation.

The user enters an equation f(x) and a value of x, and receives the EML
form, the result computed through the C VM and NumPy, and the timings.
"""

import sys
import time
import traceback

import numpy as np
import sympy as sp
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QTabWidget, QGroupBox, QGridLayout, QFrame,
    QSplitter, QMessageBox
)

from eml_core import to_eml_form, eval_eml
from eml_compiler import compile_eml, disassemble
from eml_vm import EmlVM


# ============================================================
# Styles
# ============================================================

STYLESHEET = """
QWidget {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
    color: #e0e0e0;
    background-color: #1e1e1e;
}

QTabWidget::pane {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    background-color: #252525;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #a0a0a0;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #252525;
    color: #ffffff;
    border-bottom: 2px solid #5294e2;
}

QTabBar::tab:hover {
    color: #ffffff;
}

QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #b0b0b0;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

QLineEdit {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 10px;
    color: #e0e0e0;
    selection-background-color: #5294e2;
}

QLineEdit:focus {
    border-color: #5294e2;
}

QPushButton {
    background-color: #5294e2;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 24px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #6ba3e8;
}

QPushButton:pressed {
    background-color: #3d7bc7;
}

QTextEdit {
    background-color: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 8px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    color: #d4d4d4;
}

QLabel {
    color: #c0c0c0;
}

QFrame#separator {
    background-color: #3a3a3a;
    max-height: 1px;
}
"""


# ============================================================
# Main window
# ============================================================

class EmlCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EML Calculator")
        self.setMinimumSize(700, 600)
        self.resize(800, 700)
        
        # VM initialization
        try:
            self.vm = EmlVM()
            self.vm_ok = True
        except Exception as e:
            self.vm = None
            self.vm_ok = False
            print(f"VM init error: {e}")
        
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(0)
        
        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_calc_tab(), "Evaluation")
        tabs.addTab(self._build_help_tab(), "Help")
        layout.addWidget(tabs)
    
    # ----------------------------------------------------------
    # The "Evaluation" tab
    # ----------------------------------------------------------
    
    def _build_calc_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        
        # --- Input ---
        input_group = QGroupBox("Input")
        ig_layout = QGridLayout(input_group)
        ig_layout.setSpacing(8)
        
        ig_layout.addWidget(QLabel("f(x) ="), 0, 0)
        self.input_expr = QLineEdit()
        self.input_expr.setPlaceholderText("sin(x) + log(x)")
        self.input_expr.returnPressed.connect(self._calculate)
        ig_layout.addWidget(self.input_expr, 0, 1)
        
        ig_layout.addWidget(QLabel("x ="), 1, 0)
        self.input_x = QLineEdit()
        self.input_x.setPlaceholderText("2.5")
        self.input_x.returnPressed.connect(self._calculate)
        ig_layout.addWidget(self.input_x, 1, 1)
        
        self.btn_calc = QPushButton("Evaluate")
        self.btn_calc.clicked.connect(self._calculate)
        ig_layout.addWidget(self.btn_calc, 0, 2, 2, 1)
        
        layout.addWidget(input_group)
        
        # --- Results ---
        results_group = QGroupBox("Results")
        rg_layout = QVBoxLayout(results_group)
        rg_layout.setSpacing(8)
        
        # EML form
        eml_label = QLabel("EML representation:")
        eml_label.setStyleSheet("font-weight: bold; color: #5294e2;")
        rg_layout.addWidget(eml_label)
        
        self.eml_display = QTextEdit()
        self.eml_display.setReadOnly(True)
        self.eml_display.setMaximumHeight(80)
        rg_layout.addWidget(self.eml_display)
        
        # Comparison table
        comparison_label = QLabel("Comparison:")
        comparison_label.setStyleSheet("font-weight: bold; color: #5294e2;")
        rg_layout.addWidget(comparison_label)
        
        self.comparison_display = QTextEdit()
        self.comparison_display.setReadOnly(True)
        self.comparison_display.setMaximumHeight(120)
        rg_layout.addWidget(self.comparison_display)
        
        # Bytecode
        bytecode_label = QLabel("VM bytecode:")
        bytecode_label.setStyleSheet("font-weight: bold; color: #5294e2;")
        rg_layout.addWidget(bytecode_label)
        
        self.bytecode_display = QTextEdit()
        self.bytecode_display.setReadOnly(True)
        rg_layout.addWidget(self.bytecode_display)
        
        layout.addWidget(results_group, 1)  # stretch=1
        
        return tab
    
    # ----------------------------------------------------------
    # The "Help" tab
    # ----------------------------------------------------------
    
    def _build_help_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet(
            "font-family: 'Segoe UI', sans-serif; font-size: 13px;"
        )
        help_text.setHtml("""
        <h2 style="color: #5294e2;">Input format</h2>
        
        <p>Enter a mathematical expression in the variable <b>x</b>.</p>
        
        <h3 style="color: #8ab4f8;">Functions</h3>
        <table cellpadding="4">
            <tr><td><code>sin(x)</code>, <code>cos(x)</code>, <code>tan(x)</code></td>
                <td>— trigonometry</td></tr>
            <tr><td><code>asin(x)</code>, <code>acos(x)</code>, <code>atan(x)</code></td>
                <td>— inverse trigonometry</td></tr>
            <tr><td><code>sinh(x)</code>, <code>cosh(x)</code>, <code>tanh(x)</code></td>
                <td>— hyperbolic</td></tr>
            <tr><td><code>exp(x)</code></td>
                <td>— the exponent e<sup>x</sup></td></tr>
            <tr><td><code>log(x)</code></td>
                <td>— the natural logarithm ln(x)</td></tr>
            <tr><td><code>sqrt(x)</code></td>
                <td>— square root</td></tr>
            <tr><td><code>abs(x)</code></td>
                <td>— absolute value</td></tr>
        </table>
        
        <h3 style="color: #8ab4f8;">Operators</h3>
        <table cellpadding="4">
            <tr><td><code>+</code> <code>-</code> <code>*</code> <code>/</code></td>
                <td>— arithmetic</td></tr>
            <tr><td><code>**</code></td>
                <td>— exponentiation</td></tr>
        </table>
        
        <h3 style="color: #8ab4f8;">Constants</h3>
        <table cellpadding="4">
            <tr><td><code>pi</code></td><td>— π ≈ 3.14159</td></tr>
            <tr><td><code>E</code></td><td>— e ≈ 2.71828</td></tr>
        </table>
        
        <h3 style="color: #8ab4f8;">Examples</h3>
        <table cellpadding="4">
            <tr><td><code>sin(x)**2 + cos(x)**2</code></td>
                <td>— a trigonometric identity (= 1)</td></tr>
            <tr><td><code>exp(-x) * cos(2*pi*x)</code></td>
                <td>— a damped oscillator</td></tr>
            <tr><td><code>log(x + 1) - sqrt(x)</code></td>
                <td>— a combination of functions</td></tr>
            <tr><td><code>sin(x)**3 + sinh(x)*sin(2*x)</code></td>
                <td>— a complex expression</td></tr>
        </table>
        
        <h2 style="color: #5294e2; margin-top: 20px;">What is EML?</h2>
        <p><b>EML</b> (Exponent Minus Logarithm) is a single operator:</p>
        <p style="text-align: center; font-size: 16px;">
            <code>eml(x, y) = e<sup>x</sup> − ln(y)</code>
        </p>
        <p>Through recursive nesting of eml all standard mathematical
        functions can be expressed. The application converts your expression
        into EML form, compiles it to bytecode and evaluates it on the C stack machine.</p>
        """)
        
        layout.addWidget(help_text)
        return tab
    
    # ----------------------------------------------------------
    # Evaluation logic
    # ----------------------------------------------------------
    
    def _calculate(self):
        expr_str = self.input_expr.text().strip()
        x_str = self.input_x.text().strip()
        
        if not expr_str:
            self._show_error("Enter an expression f(x)")
            return
        if not x_str:
            self._show_error("Enter a value of x")
            return
        
        try:
            x_val = float(x_str)
        except ValueError:
            self._show_error(f"Invalid value of x: '{x_str}'")
            return
        
        x = sp.Symbol('x')
        
        # --- Parsing ---
        try:
            expr = sp.sympify(expr_str, locals={'x': x})
        except Exception as e:
            self._show_error(f"Expression parsing error:\n{e}")
            return
        
        # --- Conversion to EML ---
        try:
            t0 = time.perf_counter()
            eml_expr = to_eml_form(expr)
            t_convert = time.perf_counter() - t0
        except Exception as e:
            self._show_error(f"EML conversion error:\n{e}")
            return
        
        self.eml_display.setPlainText(
            f"{eml_expr}\n\n(conversion: {t_convert*1000:.3f} ms)"
        )
        
        # --- Compilation to bytecode ---
        try:
            bytecode = compile_eml(eml_expr, x)
        except Exception as e:
            self._show_error(f"Bytecode compilation error:\n{e}")
            return
        
        self.bytecode_display.setPlainText(
            f"Instructions: {len(bytecode)}\n\n{disassemble(bytecode)}"
        )
        
        # --- Evaluation through EML (C VM) ---
        eml_result = None
        eml_time = None
        if self.vm_ok:
            try:
                eml_result, eml_time = self.vm.eval_timed(
                    bytecode, x_val, iterations=100
                )
            except Exception as e:
                eml_result = f"Error: {e}"
                eml_time = None
        else:
            eml_result = "C VM unavailable"
        
        # --- Evaluation through NumPy ---
        try:
            np_func = sp.lambdify(x, expr, modules=['numpy'])
            
            # Warm-up
            _ = np_func(x_val)
            
            t0 = time.perf_counter()
            for _ in range(100):
                np_result = np_func(x_val)
            np_time = (time.perf_counter() - t0) / 100
        except Exception as e:
            np_result = f"Error: {e}"
            np_time = None
        
        # --- Formatting the results ---
        self._display_comparison(eml_result, eml_time, np_result, np_time)
    
    def _display_comparison(self, eml_result, eml_time, np_result, np_time):
        """Formats and displays the comparison table."""
        lines = []
        
        def fmt_val(val):
            if isinstance(val, str):
                return val
            if isinstance(val, complex):
                if abs(val.imag) < 1e-15:
                    return f"{val.real:.12g}"
                return f"{val.real:.12g} {val.imag:+.12g}i"
            return f"{val:.12g}"
        
        def fmt_time(t):
            if t is None:
                return "—"
            if t < 1e-6:
                return f"{t*1e9:.1f} ns"
            if t < 1e-3:
                return f"{t*1e6:.2f} us"
            return f"{t*1000:.3f} ms"
        
        header = f"{'Method':<20} {'Result':<28} {'Time':>12}"
        lines.append(header)
        lines.append("─" * len(header))
        
        lines.append(
            f"{'EML (C VM)':<20} {fmt_val(eml_result):<28} {fmt_time(eml_time):>12}"
        )
        lines.append(
            f"{'NumPy':<20} {fmt_val(np_result):<28} {fmt_time(np_time):>12}"
        )
        
        # Speedup
        if eml_time and np_time and eml_time > 0:
            speedup = np_time / eml_time
            lines.append("")
            if speedup > 1:
                lines.append(f"EML is {speedup:.1f}x faster than NumPy")
            else:
                lines.append(f"NumPy is {1/speedup:.1f}x faster than EML")
        
        # Value difference
        if (isinstance(eml_result, (complex, float, int))
                and isinstance(np_result, (complex, float, int, np.floating))):
            diff = abs(complex(eml_result) - complex(np_result))
            if diff < 1e-12:
                lines.append(f"Difference: {diff:.2e} (match)")
            else:
                lines.append(f"Difference: {diff:.2e}")
        
        self.comparison_display.setPlainText("\n".join(lines))
    
    def _show_error(self, msg):
        """Displays an error in the results."""
        self.eml_display.setPlainText("")
        self.bytecode_display.setPlainText("")
        self.comparison_display.setPlainText(f"[ERROR]\n{msg}")


# ============================================================
# Entry point
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    
    window = EmlCalculator()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

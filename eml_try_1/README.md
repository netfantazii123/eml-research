# eml_try_1 — initial experiments with the EML operator

Sandbox for the operator `eml(x, y) = exp(x) − ln(|y| + ε)` and ways of
evaluating it quickly.

Contents:

- `eml_c.c` + `CMakeLists.txt` — the operator and whole EML graphs in C (builds
  into `eml_c.dll` for ctypes and `eml_test.exe` for standalone tests);
- `eml_core.py` — reduction of expressions to EML form and their evaluation
  (sympy);
- `eml_compiler.py`, `eml_vm.py` — compiler and VM for EML expressions;
- `main.py` — implementation benchmark: sympy vs numba vs ctypes calls vs a full
  C graph;
- `gui.py` — graphical interface to the experiments;
- `eml_base.txt` — notes on the base identities of the operator.

## Build and run

```bash
cmake -S . -B build
cmake --build build
# place build/eml_c.dll next to main.py
python main.py
```

Built binaries are not tracked.

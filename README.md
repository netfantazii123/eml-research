# EML — distilling neural network policies into analytic formulas

Train a neural "oracle" to play a game (RL), then use a genetic algorithm to
distill its policy into a compact analytic formula built on the operator

```
eml(x, y) = exp(x) − ln(|y| + ε)
```

The formula requires no torch, occupies tens of AST nodes, and runs inference on
a microcontroller (STM32/Arduino) in real time. It is also directly readable and
analyzable.

**Pipeline (3 isolated stages):**

1. **Oracle** — train a heavy neural network (GA+MLP or CNN+PPO).
2. **Distillation** — collect a `(state → oracle_action)` dataset and evolve an
   EML tree that approximates it.
3. **Inference** — export the formula to plain C/Python with no ML dependencies.

## Repository layout

| Directory | Contents |
|---|---|
| [`flappy_bird_with_ai/`](flappy_bird_with_ai/) | **Stage 1:** Flappy Bird proof of concept. Oracle is GA + a tiny MLP. The full EML concept is described in [`update.md`](flappy_bird_with_ai/update.md). |
| [`tetris/`](tetris/) | **Stage 2:** Tetris, CNN+PPO oracle with placement-based actions. Best EML formula plays at **97 % of oracle level** with 43 AST nodes and 5× lower latency. |
| [`aerial_classifier/`](aerial_classifier/) | **Stage 2.5:** the method applied to classification. 64×64 aerial imagery, classes read from dataset subdirectories. Ablation branch over a CNN embedding holds **97 %** of oracle balanced accuracy with 28 nodes; the engineered-feature branch reaches **62 %**, where GA seed spread (0.52–0.62) exceeds the effect of any individual change. |
| [`tetris_prototype/`](tetris_prototype/) | Early frame-by-frame Tetris prototype (v1). Superseded by `tetris/`. |
| [`eml_try_1/`](eml_try_1/) | Initial experiments with the EML operator: C library, expression compiler/VM, sympy vs numba vs C benchmarks. |
| `models/` | Miscellaneous checkpoints (not tracked). |

## Running

Each directory is self-contained: its own `requirements.txt` and a `main.py`
entry point. For example:

```bash
cd tetris
pip install -r requirements.txt
python main.py            # prints the command list (train / distill / bench / gui / export)
```

## Not included in the repository

Virtual environments, `*.pt` checkpoints, `*.npz` datasets, logs, and run
snapshots (`logs/`, `runs/`) are ignored — see [`.gitignore`](.gitignore). The
oracle is retrained with `python main.py train` (Tetris: ~90 min on CPU). Final
EML formulas (`*.json`), plots, and CSV benchmarks are tracked, under each
sub-project's `models/` and `results/`.

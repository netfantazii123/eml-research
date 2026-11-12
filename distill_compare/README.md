# distill_compare — classical distillation baselines for EML

An EML result says little on its own. Whether "a 20-node formula at 90 % of
the oracle" is good depends on what the established distillation methods get
from the same oracle, on the same data, under the same evaluation seeds. This
directory adds those methods, the standard benchmarks to check them on, and
runners that write rows directly comparable with the EML ones.

## Methods

All students learn from the **teacher's outputs** (logits / scores), never from
ground truth. The question is how well each model family reproduces a given
policy, not which one is the better classifier.

| name | method | what it stands for |
|---|---|---|
| `tree_d3`, `tree_d6` | CART on the teacher's argmax (behavioural cloning) | the default interpretable policy |
| `viper_d6` | VIPER (Bastani et al., 2018): DAgger + loss-weighted CART | the standard *verifiable* policy extraction method |
| `linear` | logistic regression (classify) / ridge (rank) | the simplest readable model; for Tetris it is the classic weighted afterstate evaluation |
| `kd_mlp_h16` | 16-unit MLP, Hinton et al. (2015) soft-target KD / Ba & Caruana (2014) logit matching | network-into-smaller-network compression |
| `gplearn_sr` | GP symbolic regression (gplearn, `+ − × ÷ log sqrt neg`) | an independent symbolic regression implementation |
| `classic_gp` | emlcore's GA over the control basis `+ − × ÷ exp log tanh` | same search as EML, ordinary primitives |
| `eml_data` | emlcore's GA over the EML basis, DATA phase only | EML as a pure regression student |
| `eml_task` | the task's own EML pipeline (DATA + GAME phase, VAL selection) | the thesis method itself (Flappy, Tetris) |

VIPER's weight in the paper is `V(s) − min_a Q(s, a)`. The oracles here are
policies without a Q-function (Flappy's GA-MLP has no critic), so the gap
between the teacher's best and worst logit is used instead.

## Benchmarks

* **Tabular** (`datasets.py`): banknote, pima, ionosphere, wine-quality-white
  (UCI, via a plain-CSV mirror, SHA-256-pinned) plus breast_cancer, wine, digits
  (bundled with scikit-learn). Teacher: MLP 128-64. Split 60/20/20, stratified,
  re-drawn per seed.
* **Control** (`ppo.py`, `envs.py`): CartPole-v1, Acrobot-v1, LunarLander-v3
  with PPO teachers trained by `ppo.py` (checkpoints go to `teachers/`, not tracked).
* **Project games**: Flappy Bird (the tracked `best_ga.pt` oracle) and Tetris
  (the PPO CNN oracle, `tetris/models/best_ppo.pt`, or `--oracle`). Seeds come
  from each task's own TRAIN / VAL / TEST pools.

## Metrics

* `metric`: test accuracy (tabular), episode return / pipes / lines (control).
* `pct_of_teacher`: tabular `acc / teacher_acc`; control
  `(m − random) / (teacher − random)` on the same seeds, so negative-return tasks
  are reported meaningfully.
* `fidelity`: agreement with the teacher's decision on held-out
  teacher-distribution data. `fidelity_onpolicy`: agreement on the states the
  student itself visits in the test episodes, where compounding error shows.
* `size` (nodes or parameters), `flops` (arithmetic + comparisons per decision,
  an EML node counts as exp + log + sub) and `transcendental` (libm calls).

## Running

```bash
pip install -r distill_compare/requirements.txt
python -m distill_compare.run tabular            # ~30 min on 4 CPU cores
python -m distill_compare.run gym                # trains teachers on first run
python -m distill_compare.run flappy
python -m distill_compare.run tetris --oracle tetris/models/best_ppo.pt
python -m distill_compare.report                 # results/*.csv -> REPORT.md tables
pytest distill_compare/tests
```

`--seeds`, `--methods tree_d6,linear`, `--datasets`, `--envs` and the GA
budgets (`--ga-generations`, `--eml-generations`, ...) narrow a run. Rows are
appended to `results/<suite>.csv` as they finish.

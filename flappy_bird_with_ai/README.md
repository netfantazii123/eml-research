# Flappy Bird + EML distillation (stage 1)

Proof of concept of the **NN oracle → dataset → GA evolution of an EML formula**
pipeline on a simple 2D task. The oracle is a tiny MLP trained by a genetic
algorithm; its policy is distilled into EML formulas at three complexity levels
(weak / medium / strong).

The full EML distillation concept (base operator, stability rules, role of the
stages) is described in [`update.md`](update.md).

## Running

```bash
pip install -r requirements.txt
python main.py        # command list
python main.py train  # full benchmark: GA training + EML distillation
python main.py gui    # GUI (PySide6)
python main.py demo   # pygame demo of the trained agent
```

Final EML formulas are in `models/*.json`, benchmark plots in `results/`.

## DAgger

`python dagger_distill.py --seeds 3 --iters 6` compares BC, DAgger (supervised
fit on the aggregated states the formula visits, balanced accuracy of the
action) and the existing DATA+GAME pipeline. Results go to `results/dagger_*.csv`;
the summary is in [`report/REPORT.md`](../report/REPORT.md).

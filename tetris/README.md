# Tetris + EML distillation (stage 2)

Scaling up the EML pipeline: the oracle is a CNN trained with PPO over
placement-based actions (rotation × column, 40 actions with a legality mask).
The policy is distilled into a single EML formula over engineered board features
(afterstate evaluation in the style of Dellacherie).

**Results:** the best formula plays **355.6 lines/game = 97.4 % of the oracle**
with 43 AST nodes and 5× lower latency than the CNN (103 vs 518 µs).

## Running

```bash
pip install -r requirements.txt
python main.py            # command list
python main.py train      # train the CNN oracle (PPO), ~90 min on CPU
python main.py distill    # distill the oracle into an EML formula
python main.py bench 30   # CNN vs EML comparison, CSV into results/
python main.py export     # export the formula to inline C (results/eml_formula.h)
python main.py gui        # GUI: training · distillation · play · formulas
```

Final formulas are in `models/best_eml*.json`, the microcontroller-ready C
header in `results/eml_formula.h`, and summary tables in
`results/benchmark_table.csv`. Oracle checkpoints (`*.pt`) are not tracked; they
are regenerated with `train`.

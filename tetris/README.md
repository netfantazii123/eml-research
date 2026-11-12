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

## Resuming a distillation

Every `distill` run saves its GA state under `runs/<run>/checkpoints/`, one
directory per phase (`data/`, `game/`; a batch keeps one per variant):

| File | What it is |
|---|---|
| `latest.json` | where the run stopped (every `EML_CHECKPOINT_EVERY` generations, and on stop/finish) |
| `snapshots/gen_N.json` | a resumable point every `EML_SNAPSHOT_EVERY` generations |
| `best/gen_N.json` | the full population of the generation that produced a new best (by validation); the last `EML_KEEP_BEST_STATES` are kept |
| `hall_of_fame.json` | the top `EML_HOF_SIZE` distinct formulas, with validation score and generation |

```bash
python main.py points runs/<run>                         # list points + hall of fame
python main.py distill --reuse-data --resume runs/<run>  # continue where it stopped
python main.py distill --reuse-data --resume runs/<run> --from best --extra-gens 200
python main.py distill --reuse-data --resume runs/<run> --from 300
python main.py distill --reuse-data --resume runs/<run> --extra-gens 100 --allow-new-data
python main.py distill --batch --reuse-data --resume runs/<batch run>   # a crashed batch
```

A state is saved at the top of a generation together with both random
generators, so a resumed run is bit-identical to one that was never stopped
(`tests/test_resume.py`). `--extra-gens` continues at the floor mutation
intensity (fine-tuning) rather than restarting the wild-to-fine schedule.
A checkpoint refuses to resume on a different dataset unless
`--allow-new-data` says that is the point (e.g. a DAgger-grown dataset).
From code, `evolve_formula(..., init_trees=[...])` starts a new run from a set
of formulas, e.g. `emlcore.checkpoint.load_hall_of_fame(dir, basis)`.

Speed: fitness is memoised by (formula, data) (`EML_FITNESS_CACHE`) and the
GAME phase can play its games in parallel (`EML_INGAME_WORKERS`, 0 = all
cores). Neither changes the result.

## DAgger (dataset aggregation)

`dagger_distill.py` runs DAgger on top of the DATA phase: the current formula
plays, the oracle labels every legal placement on the boards the formula
reaches, the dataset grows, the formula is refit (warm-started). It compares
BC, DAgger, BC+GAME (the existing pipeline) and DAgger+GAME on the same oracle
and the same TEST seeds. The generic loop is `emlcore/dagger.py`.

```bash
python dagger_distill.py --seeds 0 1 2 --iters 5      # -> results/dagger/
python ../report/make_report.py                        # refresh report/REPORT.md
```

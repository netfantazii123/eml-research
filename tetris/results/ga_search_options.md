# GA search options — first measurement (synthetic dataset)

Why the three options in `config.py` (`EML_UNIQUE_CHILDREN`, `EML_FOLD_CONSTANTS`,
`EML_TUNE_CONSTANTS_TOP`) are off by default: they change the search trajectory,
and on the only data available when they were written they gave no clear win.

**Setup.** No trained oracle was available, so the "oracle" is a hand-written
Dellacherie-style evaluation (holes, bumpiness, aggregate/landing height, row/col
transitions, wells, cleared lines) plus noise, collected over 20 TetrisEnv
episodes: 25,972 placements in 1,144 groups, 25 % of groups held out. DATA phase
only, population 200, 150 generations, subsample 4000, 6 GA seeds per row.
`val_fit` = regression fitness on held-out groups (higher is better);
`top1` = share of held-out steps where the formula's argmax equals the oracle's;
`evals` = distinct fitness evaluations (cache misses).

NSGA-II (the default):

| option | val_fit | top1 | nodes | evals | s |
|---|---|---|---|---|---|
| base | 1.590 ± 0.103 | 0.139 ± 0.114 | 41.7 | 31k | 11.0 |
| base, 300 gens (same evals as unique) | 1.683 ± 0.047 | 0.163 ± 0.073 | 44.3 | 63k | 20.7 |
| unique | 1.773 ± 0.052 | 0.129 ± 0.115 | 51.7 | 60k | 27.4 |
| fold | 1.491 ± 0.126 | 0.040 ± 0.026 | 36.0 | 29k | 8.2 |
| unique + fold | 1.786 ± 0.068 | 0.125 ± 0.075 | 59.0 | 60k | 26.7 |
| tune (top 5, 6 steps) | 1.565 ± 0.141 | 0.114 ± 0.090 | 39.7 | 30k | 11.7 |

Single-objective: all six variants within 1.32–1.34 val_fit; no difference.

**Reading.**
- Under NSGA-II about 40 % of all fitness requests are duplicates: environmental
  selection keeps copies of the same front point. The cache already removes
  their cost; `unique` stops them being bred, which doubles the number of
  genuinely new trees evaluated. At an equal number of distinct evaluations
  (base at 300 generations) it is still ahead on val_fit (1.77 vs 1.68,
  ~2 sd) — but not on top1, which is too noisy here to decide anything.
- Folding constants shrinks trees as expected but loses quality; tuning
  constants does nothing measurable. Neither is worth turning on as it stands.
- The synthetic oracle is not the CNN. The decision has to be made on the real
  dataset: `unique` first, against base at equal evaluations.

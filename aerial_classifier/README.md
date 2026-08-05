# aerial_classifier — distilling an image classifier into EML formulas

Applies the EML distillation method from **control** (Flappy Bird, Tetris) to
**image classification**.

Reference task: 64×64 RGB aerial imagery crops, 4 classes — `BEACH`, `MOTO`,
`PEOPLE`, `ROADS`. Dataset of 5049 images, imbalanced 3.4 : 1. The dataset is
not hard-coded: classes are read from the subdirectories of a selected folder
(see «[Data](#data)»), in any number and with any names.

## Difference from the control stages

Stages 1 and 2 were **ranking** tasks: the formula received a set of candidate
moves and picked the best one. Classification differs in two places:

1. **There are several formulas — one per class**, decision = `argmax_c f_c(x)`.
2. **Their outputs must be calibrated.** In Tetris, argmax was taken over
   different inputs to the same formula, so the common output scale cancelled.
   Here argmax is taken between four independently evolved formulas, and without
   alignment the winner is whichever formula emits larger numbers rather than
   the most appropriate one. In the first run this caused class `MOTO` to never
   be predicted at all. It is corrected by two constants per class (mean and
   standard deviation of the output on train) — on a microcontroller that is
   4 subtractions and 4 divisions.

The transfer of the method to classification is limited by the comparability of
formula scales, not by the expressiveness of the formulas. Branch C (below) is
constructed so that this problem does not arise.

## Three experiment branches

| | branch A (main) | branch B (ablation) | branch C (single formula) |
|---|---|---|---|
| oracle input | 32 engineered features | raw 64×64×3 pixels | the same 32 features |
| oracle | `FeatureMLP` (32→128→64→4) | `PixelCNN`, 3 conv blocks | the same `FeatureMLP` |
| formulas | one per class | one per class | **one for all classes** |
| decision | `argmax_c (f_c − μ_c)/σ_c` | same | threshold table over `f(x)` |
| left on the MCU | extractor + formulas | **convolutions** + formulas | extractor + formula |

Branch A is the target configuration: after distillation nothing of the neural
network remains, and only `aerial_features.h` and `aerial_eml_A.h` run on the
device. Branch B gives an upper bound on quality, but the convolutions stay on
the microcontroller, so it serves as a reference point rather than a standalone
solution. Branch C works in the same space as A and learns from the same oracle,
so the comparison does not also measure a difference between oracles.

## Features (branch A)

32 quantities computed in three passes over the frame using plain arithmetic:

* **color (14)** — RGB and luminance channel moments, opponent axes `rg`/`yb`,
  saturation;
* **texture (8)** — gradient magnitudes, edge density, anisotropy, structure
  tensor coherence, luminance entropy;
* **8×8 blocks (2)** — within-block roughness and between-block spread;
* **layout (8)** — center vs periphery, radial profile, quadrants, dark and
  bright pixel fractions.

HOG, SIFT, and FFT are deliberately excluded: they would give better
separability but no longer qualify as cheap preprocessing on an STM32.

Feature selection was measurement-driven. The first version contained
`grad_vert` (top minus bottom) and `grad_horz` (left minus right); an aerial
frame has no canonical orientation, and both scored F-statistics of 2.5 and 1.2
against 330 for the best feature. They were replaced by the rotation-invariant
`radial_slope` and `block16_spread` (F = 308 and 174). The strongest feature is
**structure tensor coherence**: roads have a pronounced dominant direction,
sand and crowds do not.

Current values: `python main.py info`.

### Scale of `center_edge_diff`

`python main.py calibrate` checks whether features occupy their allotted range.
`center_edge_diff` occupied only **24.6 %** of `[-1, 1]` (actual spread
`[-0.165, +0.327]`). It was normalized by the constant
`_SCALE_GRAD_MAG = 80.0`, i.e. the scale of the gradient itself, whereas the
feature is a *difference* of two mean gradients and is four times smaller.

This is immaterial for the oracle: an MLP has trainable weights and `BatchNorm`
and is insensitive to a linear input scale (the feature F-statistic is likewise
invariant under an affine transform — covered by a test). It matters for the EML
formula: the operator contains `exp()`, and a feature compressed into a narrow
band behaves differently from one spread across the full range.

The fix is a separate constant `_SCALE_CENTER_EDGE = 20.0`:

| | coverage of `[-1, 1]` | clipped at bound | spread |
|---|---|---|---|
| before (`_SCALE_GRAD_MAG = 80`) | 24.6 % | 0.0 % | `[-0.165, +0.327]` |
| after (`_SCALE_CENTER_EDGE = 20`) | **83.0 %** | 0.2 % | `[-0.660, +1.000]` |

The change alters the features, so it required retraining the oracle and
repeating distillation; all numbers below are recomputed after it.

**Cache invalidation.** The feature cache (`dataset/features.npz`) was validated
only by array shape `(N, 32)`. A change of scale constant does not change the
shape, so the cache would have remained "valid" and training would silently have
used the old numbers. The cache now stores a fingerprint of all constants,
thresholds, and feature names (`features.fingerprint()`) and is recomputed on
mismatch. The same applies to `dataset/cache.npz`, which previously was never
invalidated; it now checks the file count and total dataset size.

## Pipeline

```
BMP 64×64 ──► features.py (32 features) ──► oracle (MLP) ──► logits
                     │                                          │
                     │                    dataset (features → logits)
                     ▼                                          ▼
              the same 32 features ─────────► GA evolution of 4 EML formulas
                     │                                          │
                     └────────────► argmax_c (f_c − μ_c)/σ_c ◄──┘
```

Distillation runs in two phases, directly analogous to DATA/GAME from Tetris:

* **DATA phase** — symbolic regression of each formula to the logit of its
  class. Logits are z-normalized beforehand *within an example* (across the 4
  classes): this removes the arbitrary scale and preserves the argmax.
* **ACC phase** — evolution of the whole formula set, fitness = actual balanced
  accuracy. Required because a small MSE and a correct argmax are different
  things: a formula can reproduce the logits closely and still confuse classes
  at the boundary.

ACC-phase fitness is computed **on a train subsample only**. Evolving against
val is the same kind of fitting as gradient training, after which val is no
longer an independent estimate.

## Branch C: one formula for all classes

A single formula:

```
image ──► features.py (32 features) ──► oracle (MLP) ──► oracle answer
             │                                              │
             │                  dataset (features → oracle label)
             ▼                                              ▼
      the same 32 features ──────► island evolution of ONE formula
             │                                              │
             └──► v = f(features) ──► threshold table ◄──────┘
                                       (fitted TO the formula)
```

The target is the **oracle's answers**, not the ground truth labels: this
remains distillation, as in stages 1 and 2, and `fidelity` stays meaningful.
Ground truth labels are used only for reported metrics.

There is nothing to compare, no calibration constants, and the device holds one
tree and a table of a few thresholds. The scale comparability problem that
required `μ_c` and `σ_c` in branch A does not exist here.

### The threshold table cannot be fixed in advance

A fixed code such as "class 0 is 0.0, class 1 is 0.33, class 2 is 0.6, class 3
is 1.0" with the formula regressing to it fails for two independent reasons.

**The classes are not ordered.** Such a code imposes on `BEACH`, `MOTO`,
`PEOPLE`, and `ROADS` both an order and distances that they do not have. MSE
against it penalizes "distant" errors more than "near" ones, which is meaningful
only for ordinal quantities. There are twelve non-equivalent orders for four
classes, and the distances form a continuum.

**A small MSE and a correct class are different things.** The same argument that
motivates the ACC phase in branch A.

The table is therefore **fitted to the formula** at every fitness evaluation, by
exact dynamic programming over the sorted outputs, maximizing balanced accuracy.
The formula chooses in what order and at what distance to place the classes.
The resulting layout is an **output of the run**, not an assumption, and is
printed in the report:

```
class layout on the formula axis:
            -inf .. -2.526        ->  BEACH
          -2.526 .. -1.445        ->  ROADS
          -1.445 .. +2.202e+04    ->  PEOPLE
      +2.202e+04 .. +2.202e+04    ->  MOTO
      +2.202e+04 .. +inf          ->  PEOPLE
```

Testable prediction: classes adjacent on the axis should be confused more often
than others — checked against the confusion matrix in the same report.

### Multiple intervals per class instead of a periodic activation

In the table above `PEOPLE` occupies **two** intervals. More intervals than
classes are allowed (`EMLC_MAX_INTERVALS`) and labels may repeat, so a class can
occupy several zones on the axis and the linear ordering of classes stops being
a constraint.

This is the effect a periodic activation such as `sin(f(x))` would target.
Periodicity itself is not required, and would be harmful:

* the EML tree output lies roughly in `[-10, +2.2·10⁴]` (`exp(clamp(x, ±10))`
  plus a logarithm), so the data spans **thousands of sine periods**;
* the **fitness landscape becomes discontinuous**: a small edit to the tree
  shifts the output by 0.1 and half the examples jump to a different class. A GA
  does not climb such a landscape, and no number of generations corrects this;
* C computes in `float`. At an argument of ~2·10⁴ there is no precision left,
  and `verify-c` would fail on a correct export.

The threshold table provides the same expressiveness, an exact fit, and `K−1`
comparisons on the microcontroller.

### float32 resolution of the thresholds

Thresholds at the level of 2.2·10⁴ are units apart, while the `float32` step
there is about 0.002. If two thresholds converge closer than that, the interval
between them **disappears along with its class** on the device, and the C/Python
comparison fails as though the export were broken. The decoder therefore reduces
the table to one that is distinguishable in `float32` at fitting time: Python
and C operate on the same table by construction.

The divergence of the formula output between `float32` and `float64` is
measured, and it grows with tree size: `1.6·10⁻⁵` at 41 nodes, `7.1·10⁻⁵` at 39,
`6.9·10⁻⁴` at 63 — one and a half orders of magnitude at nearly the same size,
because what matters is not the node count but the depth of the `exp`/`log`
chain a value passes through. `verify-c` for branch C therefore prints and
compares the formula output as well as the class: without that quantity a
legitimate borderline divergence is indistinguishable from a broken export.

### Long runs

Branch C is designed for hours of computation, spent on **diversity** rather
than on the length of a single chain. From the branch A batch, GA spread across
seeds (0.525–0.623) exceeds the effect of any single change, so one long
evolution says little about the method.

* **Island model** — several independent populations occasionally exchanging
  their best individuals (as copies, not transfers: otherwise an island loses
  its elite and the exchange degenerates into shuffling).
* **Process-level parallelism** — a whole island goes into a process for tens of
  generations. Distributing individual fitness evaluations across processes is
  not viable: one evaluation takes about a millisecond, and transfer would cost
  more than the computation.
* **The global best is chosen by re-evaluation on the full sample.** Within an
  island fitness is computed on its own random subsample, and numbers from
  different islands are not comparable.
* **Checkpoint** every few rounds, written atomically. Only the same run is
  resumed: a checkpoint with a different number of inputs, classes, or
  individuals is rejected.

## Metrics

Three distinct quantities:

* **accuracy** — fraction of correct answers against ground truth labels;
* **fidelity** — fraction of matches with the oracle's answer (whether the
  formula reproduces the oracle specifically);
* **balanced accuracy** — accuracy averaged over classes. The primary metric: at
  a 3.4 : 1 imbalance, plain accuracy gives 0.25 balanced for the degenerate
  "everything is `ROADS`" while reaching 0.41 plain.

A formula can have high accuracy and low fidelity, meaning it solves the task in
its own way rather than copying the oracle. Both quantities are therefore
computed.

**Oracle checkpoint selection also uses balanced accuracy.** The loss is
class-weighted and the primary metric is balanced, so early stopping, LR decay,
and best-checkpoint selection use balanced accuracy as well; previously they
used plain accuracy. At a 3.4 : 1 skew, plain accuracy pulls selection in the
direction the weighted loss moves away from: an epoch that predicts the frequent
`ROADS` better at the cost of the rare `BEACH` wins on plain accuracy and loses
on balanced. Both curves are written to the training history; the distance
between them measures the skew. Every table below states which metric is shown.

## Results

All numbers below were obtained **after** the `center_edge_diff` scale fix and
the switch of checkpoint selection to balanced accuracy: both oracles retrained,
distillation run as a batch of 5 variants per branch.

### Oracles

| | model | parameters | val acc | val bal | test acc | test bal | training |
|---|---|---|---|---|---|---|---|
| branch A | `FeatureMLP` | 13 124 | 0.9037 | 0.9082 | 0.8970 | **0.9053** | **14 s** |
| branch B | `PixelCNN` | 292 164 | 0.9142 | 0.9202 | 0.9287 | **0.9256** | 3087 s |

Before the changes (plain accuracy): branch A — val 0.880, test 0.889; branch B
— val 0.913, test 0.929. The branch A oracle gained ~1.5 points; the branch B
oracle did not change, as expected — it consumes raw pixels and does not see
the engineered features, and balanced-metric selection had almost no effect
on it.

32 hand-written features with a small MLP trail the convolutional network by
**2 points** of balanced accuracy while training 220× faster.

### Distillation

Metrics on test, **balanced**. The formula set was selected **by val** out of
five runs with different seeds and penalties:

| | oracle | EML | fraction of oracle | fidelity | nodes | inputs |
|---|---|---|---|---|---|---|
| branch A | 0.9053 | 0.5571 | **61.5 %** | 0.5152 | 172 | 32 / 32 |
| branch B | 0.9256 | 0.9009 | **97.3 %** | 0.9445 | 28 | 11 / 32 |

**Branch A did not improve.** The spread across five seeds exceeded the effect
of the changes:

| seed | penalty | val bal | test bal | nodes |
|---|---|---|---|---|
| 0 | weak | **0.5807** ← selected | 0.5571 | 172 |
| 1 | medium | 0.5675 | 0.6097 | 30 |
| 2 | strong | 0.5277 | 0.5246 | 8 |
| 3 | weak | 0.5524 | 0.5856 | 156 |
| 4 | medium | 0.5709 | **0.6229** | 78 |

The previously reported value of 0.611 lies inside this spread (0.525–0.623),
i.e. it was one point from a distribution rather than a result attributable to
the state before the scale fix. A single GA run carries little information about
the method, and comparing single "before" and "after" runs is not meaningful.

Selection by val picked the **worst set on test** of the five (0.5571 against
0.6229 for the best) and also the largest, 172 nodes against 30. With 758
validation examples and a 10-point spread, val-based selection is as noisy as
what it selects. The protocol is nonetheless respected — test must not be used
for selection.

**Branch B improved:** 90.5 % → **97.3 %** of the oracle, 22 → 28 nodes,
fidelity 0.857 → 0.944. The feature scale fix cannot be responsible (branch B
does not use the features), leaving checkpoint selection by balanced metric and
the batch of five seeds instead of a single run. The spread here is three times
smaller than in branch A: test bal 0.8905–0.9089.

Formulas over the CNN embedding hold 97 % of the oracle with 28 nodes and 11
inputs; over engineered features, 62 % with 172 nodes and all 32 inputs. The
cause is not the features but which component does the nonlinear work: the CNN
is trained so that its embedding is separable by a **linear** head, leaving the
formula to approximate a nearly linear function. In branch A the tree of depth
≤ 7 must supply all the nonlinearity, which is insufficient.

The more linearized the input to the formula, the cheaper the distillation. In
branch B this is paid for with the convolutions that remain on the device.

### Branch C: batch of five seeds

Cap of 3000 generations, population 400, 8 islands on 8 processes. All five runs
stopped early on `patience` — between 925 and 2975 generations, 16 to 70
minutes. Metrics on test, **balanced**; the set was selected **by val**.

| seed | penalty | generations | val bal | test bal | nodes | intervals |
|---|---|---|---|---|---|---|
| 0 | weak | 1100 | 0.6042 | 0.5944 | 69 | 4 |
| 1 | medium | 1900 | **0.6332** ← selected | 0.6237 | 61 | 4 |
| 2 | strong | 925 | 0.5632 | 0.5664 | **23** | 4 |
| 3 | weak | 2375 | 0.5971 | 0.6233 | 95 | 6 |
| 4 | medium | 2975 | 0.6181 | **0.6305** | 59 | 4 |

Comparison with branch A on the same oracle and the same features:

| | formulas | test bal (5 seeds) | mean | nodes | inputs | constants |
|---|---|---|---|---|---|---|
| branch A | 4 | 0.5246 – 0.6229 | 0.5800 | 8 – 172 | 32 / 32 | 8 (μ and σ) |
| branch C | **1** | 0.5664 – 0.6305 | **0.6077** | 23 – 95 | 16 / 32 | 4 thresholds |

**One formula is not worse than four.** The mean test value is 0.028 higher at a
spread of 0.064, i.e. borderline significant; val-based selection did not
mispick here, and the selected formula holds **69.2 %** of oracle balanced
accuracy with 61 nodes, against 61.5 % with 172 nodes for branch A — three times
more compact and eight points closer to the oracle.

**The spread is narrower.** 0.064 against 0.098 on test. This follows partly
from the island model (eight populations instead of one chain) and partly from
not having to reconcile four independently evolved scales.

**The class layout on the axis does not reproduce across seeds.** On a short
batch (200 generations, for budget comparison) five seeds gave five different
orders:

```
seed 0   ROADS < BEACH < MOTO < PEOPLE
seed 1   BEACH < MOTO < PEOPLE < ROADS
seed 2   BEACH < MOTO < ROADS < PEOPLE
seed 3   PEOPLE < MOTO < PEOPLE < ROADS < BEACH < ROADS
seed 4   ROADS < BEACH < PEOPLE < MOTO
```

The layout is a property of the particular tree found, not of the data
structure. Quality reproduces; placement does not.

The freedom to occupy several zones is used in practice: seed 3 has six
intervals, with `PEOPLE` and `ROADS` holding two each.

Compute budget: the short batch (200 generations, ~3 minutes per run) gave
0.5664–0.5916 at a mean of 0.5783 — nine tenths of the result for a tenth of the
time. Long runs buy the last two or three points, not an order of magnitude.

### Different dataset: 2 classes

The same code on a different folder — `PEOPLE` and `ROADS`, two classes, the
same feature extractor and the same oracle type:

| | oracle | EML | fraction of oracle | fidelity | nodes | intervals |
|---|---|---|---|---|---|---|
| branch C, 2 classes | 0.9394 | 0.7892 | **84.0 %** | 0.8177 | 41 | 2 |
| branch C, 4 classes | 0.9017 | 0.6237 | 69.2 % | 0.6473 | 61 | 4 |

1350 generations, 21 minutes, 12 inputs out of 32. The threshold table
degenerates to a single threshold, as expected on a binary task:

```
            -inf .. +1.858        ->  PEOPLE
          +1.858 .. +inf          ->  ROADS
```

The branch C ceiling on aerial imagery is set by the task rather than the
method: four classes in which `MOTO` and `PEOPLE` are systematically confused
even by the oracle are a hard case for a one-dimensional projection, whereas on
a separable pair of classes the same 41-node formula reproduces the network on
82 % of answers. The four-class and two-class numbers are **not comparable**
(balanced accuracy on two classes starts at 0.5); they are shown together to
locate the limit of the method.

### Effect of the size penalty

The first version penalized depth only, as in Tetris. Depth is bounded by
`EML_MAX_DEPTH`, so within the limit the tree grew in width at no cost; the node
count term (`EML_SIZE_PENALTY_WEIGHT`) addresses this. The branch A batch (the
same five runs as above, grouped by penalty):

| penalty | nodes | test bal.acc |
|---|---|---|
| `weak` (0.01) | 156, 172 | 0.586, 0.557 |
| `medium` (0.05) | 30, 78 | **0.610, 0.623** |
| `strong` (0.15) | 8 | 0.525 |

A weak penalty gives trees 5× larger and a worse result, a strong one truncates
to eight nodes and loses more; the best result is in between. 172 nodes lose to
thirty.

### Speed and size

Measurement from `python main.py bench` (numpy/torch on CPU, batched):

```
feature extractor    :    508.7 µs/frame
MLP oracle (branch A):      0.7 µs/frame
CNN oracle (branch B):   1628.1 µs/frame
EML formulas         :      3.6 µs/frame  (172 nodes)
```

**The formulas are not faster than the MLP** — on numpy batches a small matrix
multiply is cheaper than a tree traversal. With the selected 172-node set they
are five times slower, and frame time in branch A is 99 % determined by the
feature extractor (509 of 512 µs). The branch A advantage is in three other
properties:

* **size**: 13 124 MLP parameters are ≈ 51 KB of float32 weights, while the
  formulas are 5.5 KB of generated C including comments and zero weights
  (branch B: 292 164 parameters ≈ 1.1 MB, with 4.1 KB of formulas — plus the
  convolutions themselves);
* **dependencies**: no torch, no runtime — only `<math.h>`;
* **transparency**: `f_MOTO = eml(center_edge_diff, eml(eml(center_gray_diff,
  −0.792), grad_y))` is readable and verifiable by inspection.

Not all formulas are readable. `f_MOTO` above is three nodes, while `f_BEACH` of
the same set is a 172-node tree that cannot be checked by inspection.
Transparency here is a consequence of the size penalty, present where the
formula stayed small. Branch C is more favorable: a single tree, and the
val-selected variant of the batch is 61 nodes for all four classes against 172
for branch A.

The direct speed comparison is branch A as a whole (512 µs) against the CNN
(1628 µs): **3.2×**.

### C export verification

`python main.py verify-c --branch A` and `--branch C` on 200 test frames:

```
branch A (4 formulas)                    branch C (one formula, 41 nodes)
max feature divergence : 9.13e-07        max output divergence  : 1.62e-05
class agreement        : 200/200         max feature divergence : 8.80e-07
PASSED                                   class agreement        : 200/200
                                         PASSED
```

## Running

```bash
cd aerial_classifier
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

python main.py --dataset <folder> info  # select dataset (remembered)
python main.py info                     # dataset and feature F-statistics
python main.py train --branch A         # oracle (14 s on CPU; branch B — 50 min)
python main.py distill --branch A       # GA distillation (~40 min per run)
python main.py distill --branch C       # one formula, islands, one per core
python main.py export --branch C        # C code into results/c_export/
python main.py verify-c --branch C      # compile C and compare against Python
python main.py bench                    # speed comparison
```

`python main.py distill --batch` runs several variants with different seeds and
complexity penalties; the best **by val** is saved as the main one.

Branch C options:

```bash
python main.py distill --branch C --islands 8 --workers 8   # 0 = one per core
python main.py distill --branch C --generations 3000        # long run
python main.py distill --branch C --restart                 # do not resume saved state
python main.py distill --branch C --batch --variants 5 --generations 200
```

A branch C run writes state to `runs/checkpoints/eml_C_seed<N>.json` and resumes
an interrupted run by default. After normal completion the checkpoint is
deleted; otherwise the next run with the same seed would silently continue an
already finished one and immediately hit `patience`.

Any run — single or batch — is first written in full to `runs/`, and only then
compared by val against the saved `models/best_eml_A.json`. A worse run does not
overwrite it (the previous file is kept alongside as `.prev.json`), and a
manually interrupted run never claims best. Previously any run with `save=True`
(the default in both GUI and CLI) overwrote the batch-selected result.

## Model library

Every training and every distillation places a named copy into
`models/library/`, while the active model — the one used by C export,
`verify-c`, `bench`, and the test tabs — stays in place (`models/oracle_A.pt`,
`models/best_eml_A.json`).

```bash
python main.py models                             # list with characteristics
python main.py train --branch A --name "mlp-32features"
python main.py distill --branch A --name "medium-seed4"
python main.py models --activate mlp-32features   # select the active one
python main.py models --rename old new
python main.py models --delete unwanted
```

```
   kind    branch name                      val bal  test bal   complexity
★• oracle  A      mlp-features-bal0.908      0.9082    0.9053  13124 par.
   oracle  A      fast-12epochs              0.8017    0.8138  13124 par.
★• formula B      batch-seed1-medium-28nodes 0.8958    0.9009     28 nodes
```

`★` — best in its group, `•` — currently active. The marks are independent: the
active model need not be the best one.

Three rules:

* **Best is determined by val**, not by test. Test is inspected once, at the
  end; selecting a model by it would mean looking into the held-out sample.
* **A taken name is not overwritten** — it is qualified to `name-2`. Runs take
  from seconds to an hour.
* **Training always occupies the active slot** (after `train` the slot must hold
  the model that was just trained, otherwise the command misreports its action),
  but if it is worse than the best saved one this is stated explicitly, with a
  ready revert command. Distillation is stricter: it does not touch the active
  slot if the run is worse by val.

The same is available in the GUI on the **Models** tab, and the test tabs have a
model dropdown, so one frame can be run through two different formula versions.

## Tests

```bash
pip install pytest
python -m pytest                 # 211 tests, ~4 s
```

Each run prints the 15 slowest tests with their times and the final
`211 passed in Xs` (configured in `pytest.ini`).

The tests need neither the dataset nor trained checkpoints — everything required
is generated in the tests themselves (BMP and PNG are assembled byte by byte,
trees and inputs are random). The suite therefore runs on any machine. Without
torch installed, two files are skipped (`test_pipeline.py` and
`test_library.py`, which work with checkpoints); the rest run as usual.

| file | coverage |
|---|---|
| `test_data.py` | BMP parser: both row orders, row padding to 4 bytes (at a width not divisible by 4 the image would shear), BGR→RGB channel order; classes from subdirectories and alphabetical order; cache invalidated on a change of the class SET, not only of the file count; split — partition without overlap or loss, determinism, preservation of class proportions |
| `test_imaging.py` | PNG in all five row filters (the project's own encoder writes only filter zero, so the Sub/Up/Average/Paeth branches are never exercised on project files); downscaling averages rather than decimates; conversion does not stretch the image |
| `test_features.py` | all 32 features in `[-1, 1]` and finite on any image; `radial_slope` and `coherence` are invariant to rotation and reflection (feature selection relies on this); the cache fingerprint changes when any constant is edited |
| `test_eml.py` | `to_dict`/`from_dict` round-trip without loss; `evaluate` and `evaluate_batch` agree; a degenerate formula gets scale 1; calibration prevents a degenerate argmax; balanced accuracy against a manually computed confusion matrix |
| `test_eml_single.py` | the decoder DP finds the optimum (checked against exhaustive search); the promised score equals the actual accuracy of the same decoder; a class receives several zones when that is advantageous; thresholds stay distinguishable in `float32`; a checkpoint resumes a run, and migration does not deplete the donor island |
| `test_vision.py` | tiling: exact pixel match after the "mosaic → windows" round trip, empty list on a frame smaller than the window; the overlay does not corrupt the source frame; classes of a foreign dataset get distinct colors rather than a shared gray |
| `test_export.py` | no unfilled placeholders remain in the C template, all 32 features are filled, `f[26]` uses its own scale constant; the branch C threshold table reads back from C as the same `float32`, and a degenerate table does not produce a zero-length array |
| `test_pipeline.py` | a run worse than the saved one does not become active; an interrupted run never claims best; every run survives in `runs/`; a branch C formula is restored from JSON together with its threshold table; a model with a foreign class set is not applied silently |
| `test_library.py` | a user-supplied name does not break the path or escape the directory; a name collision does not overwrite another run; renaming does not lose the active mark; deletion from the library does not touch the active slot |

Two integration checks require the dataset and models: `python vision.py`
(tiling and engine accuracy for all branches on a mosaic) and
`python main.py verify-c --branch A|C` (C compilation and comparison against
Python).

## Graphical interface

```bash
python main.py gui              # or: python main.py gui --tab test
```

Six tabs: **Data**, **Features**, **Training**, **Distillation**, **Models**,
**Test**. Under each plot there is a "Save plot" button; the file is written to
`results/`.

The **Data** tab selects the folder of classes. Its contents are shown **before**
applying — classes with counts, formats, sizes — with a separate warning when
the class set differs from the current one, since trained models will not apply
to it.

The **Distillation** tab switches between branches. For branch C the "ACC phase"
checkbox is replaced by its own controls — islands, processes, run resumption —
and the formula panel prints the class layout on the axis below the formula.

Training and distillation show **live curves** and stop on a "Stop" button:
stopping happens at the nearest epoch or generation and saves the best found so
far rather than discarding the work.

Only one heavy task can run at a time: training rewrites `models/oracle_X.pt`
and distillation reads it, so running them together would read the checkpoint
mid-write. The tab that starts work takes a shared lock, and the others disable
their start buttons for the duration.

### Test · Photo

Classifies a single crop and shows the answers of **both engines side by side**:
formulas against the oracle, with per-class scores after softmax and a mark
where they diverge. The source is a random image from the dataset (with a class
filter, which also shows the ground truth label) or a file from disk.

The numbers next to the answers are labeled differently. For the oracle, the
softmax of the logits is an actual class probability estimate. For branch A and
B formulas the outputs were never calibrated as probabilities, and their softmax
is only a monotone mapping to `[0, 1]`, so it is labeled `score`, with a note
that with four classes the softmax maximum is never below `0.25` — a confidence
threshold below that value filters nothing.

For branch C, softmax is unusable: its scores live on the formula output axis
(up to 2.2·10⁴) and saturate softmax to one almost always. Instead the **margin
to the threshold** is shown — how deep the frame sits inside its interval of the
axis: `0` — at the boundary, `1` — exactly in the middle. The scale reaches
zero, so a confidence threshold on video is meaningful.

An image of arbitrary size is center-cropped to 64×64 rather than stretched:
stretching would change the texture scale that `coherence`, `block_std`, and
other features rely on.

### Test · Video

The frame is cut into 64×64 windows — the size the model was trained on — each
window is classified, and a semi-transparent fill in the class color is overlaid.
Window step, transparency, and confidence threshold are configurable (windows
below the threshold are not filled, showing where the model is uncertain). FPS
and inference time are displayed; the result can be written to mp4.

Three sources:

| source | purpose |
|---|---|
| video file | mp4/avi/mov, played in a loop |
| webcam | live stream check |
| **synthetic from dataset** | mosaic of test crops |

In the synthetic source every window has a **known ground truth label**, so the
corner shows the actual labeling accuracy. This is the way to check the video
mode numerically without aerial footage at hand. Measurement over 8 frames of 84
windows each:

| engine | labeling accuracy | inference |
|---|---|---|
| EML, branch A | 52.4 % | 44 ms/frame |
| oracle, branch A | 85.3 % | 64 ms/frame |
| EML, branch B | 88.1 % | 140 ms/frame |
| oracle, branch B | 89.6 % | 142 ms/frame |

Each row agrees with the accuracy of the corresponding model on test, i.e.
tiling loses and shifts nothing.

The model is trained on aerial imagery. On webcam video it produces confident
nonsense — operation outside the training distribution, not a malfunction.

### Implementation notes

Training, distillation, and video decoding run in separate threads, and results
are passed to the interface through `queue.Queue`. Overlay settings are read in
the main thread into a plain dict (`_refresh_settings`) and only then read by
the worker: tkinter is not thread-safe, and reading a `StringVar` from another
thread raises `RuntimeError: main thread is not in main loop` intermittently.

Frames are displayed through `tk.Canvas` and `PhotoImage` from manually
assembled PPM, so image display requires neither Pillow nor OpenCV. OpenCV is
needed **only** for the video tab (decoder) and for reading jpeg; bmp and png are
read by the project's own parsers in `imaging.py`. Without OpenCV everything
else works, and the video tab reports what is missing.

## Data

**Classes are subdirectories.** No hard-coded names: a folder is selected, its
subdirectories become the classes, and their names become the class names.

```
<any folder>/
  BEACH/*.bmp      MOTO/*.png      PEOPLE/*.jpg      ROADS/...
```

```bash
python main.py --dataset C:\path\to\folder info  # the choice is remembered
python main.py gui                               # or via the Data tab button
```

Class order is alphabetical: the class index enters both the oracle weights and
the branch C threshold table, so it cannot depend on the order in which the OS
returned the directory entries. Subdirectories without images are skipped, since
a class without examples would break both stratification and recall.

Formats and sizes:

| format | reader |
|---|---|
| `.bmp` | own numpy parser (`imaging.py`), 24-bit BI_RGB |
| `.png` | own `zlib` decoder, all five row filters |
| `.jpg`, `.jpeg` | only through OpenCV or Pillow, if installed |

An image that is not 64×64 is center-cropped to a square and scaled by **area
averaging**, not stretched or decimated. Stretching would change the texture
scale along one axis — exactly what `anisotropy` and `coherence` measure;
decimation would turn a regular texture into moiré, and the texture features
would measure a scaling artifact.

The first run caches the decoded array next to the dataset (`cache.npz`, ~62 MB
on the reference set). The cache fingerprint includes the **class names**, file
count, and total size: without the names, switching to a folder with the same
file count would reuse the old cache and the labels would mean something
different.

**Changing the dataset makes old models inapplicable,** and this is an explicit
refusal: the class list is written into every checkpoint and every formula JSON
and is verified on load. Such models are marked `≠` in the library, do not take
part in best-model selection (their metrics are not comparable: on two classes
balanced accuracy starts at 0.5), and `bench` and `vision.py` skip them with a
"skipped" line rather than failing.

```
★•  formula C  20260804-1352_bal0.821    0.8213  0.7892   41 nodes
  ≠ formula A  batch-seed0-weak-172nodes 0.5807  0.5571  172 nodes
```

The reference dataset (83 MB) is not part of the repository — see `.gitignore`.

## Microcontroller export

`python main.py export` generates self-contained headers (only `<math.h>`):

* `aerial_features.h` — extractor of 32 features from a `uint8_t[64*64*3]`
  buffer;
* `aerial_eml_A.h` — 4 formulas, calibration constants, and `aerial_classify()`;
* `aerial_eml_C.h` — **one** formula, threshold table, and `aerial_classify()`:
  `K−1` comparisons and no divisions.

In Tetris only the formula was exported: features were supplied by the game
itself. Here the extractor is the entire preprocessing step, and without it the
claim that the neural network is no longer needed is unsupported.

`python main.py verify-c --branch A|C` compiles the generated code with gcc,
runs test frames through it, and compares against Python — features
element-wise and the resulting class.

The feature tolerance is strict (`1e-4`), while class divergence is tolerated
deliberately: C computes in `float` and Python in `float64`, so on a frame where
the decision is marginal the answers may legitimately differ. Requiring exact
1.0 class agreement produced failures on a correct export. What counts as
marginal differs by branch:

* **A and B** — the gap between the two best calibrated scores is below `1e-3`;
* **C** — the distance from the formula output to the nearest threshold is no
  greater than the `float32`/`float64` divergence on that frame. The C harness
  therefore prints the formula output as well as the class: the divergence
  varies by one and a half orders of magnitude across formulas
  (`1.6e-05` … `6.9e-04`).

Branch C thresholds are written with nine significant digits — the minimum at
which the decimal form uniquely recovers the `float32`. The threshold is the
decision boundary, and `%.6f` would round it to a neighboring representable
number.

## Layout

| file | role |
|---|---|
| `config.py` | hyperparameters, feature names, dataset selection |
| `imaging.py` | BMP and PNG decoders, conversion of an image to model input |
| `data.py` | class folder scanning, cache, stratified split |
| `features.py` | extractor of 32 features |
| `oracle.py` | `FeatureMLP` (branches A and C) and `PixelCNN` (branch B) |
| `train_oracle.py` | oracle training |
| `eml_distiller.py` | EML AST, GA, DATA and ACC phases, `EMLClassifier` (A and B) |
| `eml_single.py` | branch C: threshold table, island evolution, `EMLSingle` |
| `pipeline.py` | logit dataset collection, full run, metrics |
| `export.py` | C generation |
| `reports.py` | plots and speed measurements |
| `storage.py` | oracle checkpoints, formula JSON, named model library |
| `vision.py` | test engine: frame tiling, overlay, video sources |
| `gui.py` | tkinter interface |
| `main.py` | CLI |
| `tests/` | 211 pytest tests, without dataset or torch |

`vision.py` is separated from `gui.py` so that frame classification logic does
not depend on tkinter, and can be run from the console and checked numerically:

```bash
python vision.py        # self-check: tiling, accuracy of both engines, FPS
```

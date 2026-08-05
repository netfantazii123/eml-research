# EML distillation — concept

Neuro-symbolic control: compressing the logic of a trained network into a
compact analytic formula that runs on low-power microcontrollers in real time,
and that can be read and analyzed directly.

## 1. Mathematical basis

The binary Exp-Minus-Log (EML) operator:

$$
eml(x, y) = \exp(x) - \ln(|y| + \epsilon)
$$

The absolute value and **$\epsilon$** are required to guard against the
logarithm singularity **$\ln(\le 0)$** during evolution of the formula.

## 2. Pipeline

Three isolated stages:

1. **Oracle training (reinforcement learning).** Train a heavy conventional
   network (MLP/PPO) to control the object in a simulator. Result: a black-box
   behavior model.
2. **Symbolic distillation (neuro-symbolic step).** Collect an "environment
   state **$\to$** oracle action" dataset. Use a genetic algorithm (together with
   gradient descent for weight fitting) to build an EML tree approximating the
   oracle's answers on that data.
3. **Inference.** Emit the resulting formula (AST) as C++/Python code for direct
   control without a neural network.

## 3. Stage 1 configuration (2D — Flappy Bird)

* **Environment:** Flappy Bird clone (Pygame).
* **Oracle:** a small network trained to play the game. Input is 3–4 parameters
  (bird y coordinate, y velocity, distance to pipes).
* **Distillation:** the network logic becomes an EML equation of the form
  `jump = eml(eml(y, speed), dist)`.
* **Metrics:** FPS, RAM consumption, and evaluation latency of PyTorch/TFLite
  inference against direct evaluation of the EML formula.

## 4. Engineering constraints

1. **EML is stateless.** The formula has no memory. Derivatives (rates of
   change, deltas) are **always** computed outside, in the surrounding code, and
   passed into the EML function as separate arguments.
2. **Depth penalty (floating-point drift).** Deep operator nesting accumulates
   `float` rounding error exponentially. The genetic algorithm's fitness
   function must penalize trees for excess depth levels; the solution has to
   stay flat.
3. **Gradient explosion.** When gradient descent is used for training, network
   outputs must pass through a normalization/clamping function (e.g. `Tanh`)
   before the EML layer, so that **$\exp(x)$** does not produce `NaN`.

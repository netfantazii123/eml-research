"""
GANet — the network for the genetic algorithm plus its trainer.
Architecture: 4 -> 16 -> 1 (sigmoid). 97 parameters.

IMPORTANT: train_ga() reads its hyperparameters from the `config` module at call
time rather than from defaults captured at import time — so that config changes
made from the GUI/tests actually take effect.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from config import STATE_SIZE
from env import FlappyEnv


class GANet(nn.Module):
    """
    A compact network for the genetic algorithm.

    Architecture: 4 -> 16 -> 1 (sigmoid).
    It is not trained by gradients — the weights evolve through mutation.
    Action: sigmoid(output) > 0.5 => flap, otherwise no-flap.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns the flap probability (0..1)."""
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def get_action(self, state: np.ndarray) -> int:
        """Choose an action from the state (without gradients)."""
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0)
            prob = self.forward(t).item()
        return 1 if prob > 0.5 else 0

    def get_sigmoid(self, state: np.ndarray) -> float:
        """Return the raw sigmoid output (for distillation)."""
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0)
            return self.forward(t).item()

    def mutate(self, rate: float = 0.1, std: float = 0.3):
        """In-place weight mutation."""
        with torch.no_grad():
            for param in self.parameters():
                mask = torch.rand_like(param) < rate
                noise = torch.randn_like(param) * std
                param.add_(mask * noise)

    def copy_from(self, other: "GANet"):
        """Copy the weights from another network."""
        self.load_state_dict(other.state_dict())


def evaluate_agent(agent, env: FlappyEnv,
                   max_frames: int | None = None) -> float:
    """
    Run one agent through one game and return its fitness.
    Fitness = score * 100 + frame_count.
    """
    if max_frames is None:
        max_frames = config.GA_MAX_FRAMES
    env.reset()
    frames = 0
    while not env.done and frames < max_frames:
        action = agent.get_action(env.get_state())
        env.step(action)
        frames += 1
    return env.score * 100.0 + frames


def train_ga(
    generations: int | None = None,
    population_size: int | None = None,
    elitism: int | None = None,
    mutation_rate: float | None = None,
    mutation_std: float | None = None,
    max_frames: int | None = None,
    patience: int | None = None,
    target_score: int | None = None,
    verbose: bool = True,
    should_stop=None,
) -> dict:
    """
    Training by neuroevolution (a genetic algorithm).

    Hyperparameters: taken from the current `config` when not passed.

    Auto-stop:
        - patience generations without best_fitness improvement -> stop.
          patience=0 disables the plateau stop.
        - target_score > 0 -> stop once that score is reached.
        - should_stop() returning True -> stop (for cancellation from the GUI).

    Returns:
        dict: 'best_agent', 'history', 'total_frames', 'elapsed',
              'stopped_early', 'stop_reason'.
    """
    # Resolve the parameters from config (at call time, not at import time)
    if generations is None:    generations = config.GA_GENERATIONS
    if population_size is None: population_size = config.GA_POPULATION
    if elitism is None:        elitism = config.GA_ELITISM
    if mutation_rate is None:  mutation_rate = config.GA_MUTATION_RATE
    if mutation_std is None:   mutation_std = config.GA_MUTATION_STD
    if max_frames is None:     max_frames = config.GA_MAX_FRAMES
    if patience is None:       patience = getattr(config, 'GA_PATIENCE', 0)
    if target_score is None:   target_score = getattr(config, 'GA_TARGET_SCORE', 0)

    env = FlappyEnv()
    population = [GANet() for _ in range(population_size)]
    history = []
    total_frames_all = 0

    best_fitness_ever = -float('inf')
    best_agent_ever: GANet | None = None
    patience_counter = 0
    stop_reason = ''

    t0 = time.perf_counter()

    for gen in range(generations):
        if should_stop is not None and should_stop():
            stop_reason = 'cancelled'
            if verbose:
                print(f"  -> Cancelled at gen {gen}.")
            break

        fitness_scores = []
        for agent in population:
            f = evaluate_agent(agent, env, max_frames)
            fitness_scores.append(f)
        total_frames_all += int(sum(fitness_scores))

        ranked = sorted(zip(fitness_scores, population),
                        key=lambda x: x[0], reverse=True)
        fitness_scores = [f for f, _ in ranked]
        population = [agent for _, agent in ranked]

        best_fitness = fitness_scores[0]
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        best_score = int(best_fitness // 100)

        # ── Auto-stop: plateau tracking ───────────────────────────────────
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            snapshot = GANet()
            snapshot.copy_from(population[0])
            best_agent_ever = snapshot
            patience_counter = 0
        else:
            patience_counter += 1

        gen_stats = {
            'generation': gen,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'best_score': best_score,
            'patience': patience_counter,
        }
        history.append(gen_stats)

        if verbose:
            tag = f" | plateau {patience_counter}/{patience}" if patience > 0 else ''
            print(f"  Gen {gen:3d} | "
                  f"Best fitness: {best_fitness:8.0f} | "
                  f"Avg: {avg_fitness:8.0f} | "
                  f"Best score: {best_score:3d}{tag}")

        # ── Stopping conditions ───────────────────────────────────────────
        if target_score and target_score > 0 and best_score >= target_score:
            stop_reason = f'target_score>={target_score}'
            if verbose:
                print(f"  -> Early stop: target score {target_score} reached "
                      f"at gen {gen}.")
            break

        if patience > 0 and patience_counter >= patience:
            stop_reason = f'plateau_{patience}'
            if verbose:
                print(f"  -> Early stop: no improvement for "
                      f"{patience} generations.")
            break

        # ── New generation ───────────────────────────────────────────────
        new_population = []
        for i in range(elitism):
            elite = GANet()
            elite.copy_from(population[i])
            new_population.append(elite)

        while len(new_population) < population_size:
            idx = np.random.choice(len(population), size=3, replace=False)
            best_idx = max(idx, key=lambda i: fitness_scores[i])
            child = GANet()
            child.copy_from(population[best_idx])
            child.mutate(rate=mutation_rate, std=mutation_std)
            new_population.append(child)

        population = new_population

    elapsed = time.perf_counter() - t0
    # The best agent seen overall, not merely the best of the last generation
    best_agent = best_agent_ever if best_agent_ever is not None else population[0]

    return {
        'best_agent': best_agent,
        'history': history,
        'total_frames': total_frames_all,
        'elapsed': elapsed,
        'stopped_early': bool(stop_reason),
        'stop_reason': stop_reason,
    }

"""
Flappy Bird AI — neuro-symbolic control through EML distillation.

Entry point:
    python main.py gui       — graphical interface (PySide6)
    python main.py train     — full benchmark (GA + 3 EML variants)
    python main.py demo      — pygame demo
    python main.py test      — smoke test of the modules
"""

import sys
import time
import numpy as np


def cmd_train():
    """Full benchmark."""
    from benchmark import run_benchmark
    run_benchmark(verbose=True)


def cmd_demo():
    """Pygame demo."""
    from demo import run_demo
    run_demo()


def cmd_gui():
    """Graphical interface."""
    from gui import main as gui_main
    gui_main()


def cmd_test():
    """Smoke test of all modules."""
    from env import FlappyEnv
    from ga_net import GANet, train_ga
    from eml_distiller import (
        EMLNode, eml_op, make_basis,
        collect_dataset, evolve_eml, EMLAgent,
    )
    from emlcore import search

    def _check(cond: bool, msg: str):
        if not cond:
            raise AssertionError(msg)

    # ── 1. FlappyEnv ────────────────────────────────────────────────────
    print("=" * 50)
    print("   FlappyEnv Smoke Test")
    print("=" * 50)

    env = FlappyEnv()
    episodes = 1000
    total_frames = 0
    t0 = time.perf_counter()
    for _ in range(episodes):
        env.reset()
        while not env.done:
            env.step(np.random.randint(0, 2))
            total_frames += 1
    elapsed = time.perf_counter() - t0
    fps = total_frames / elapsed

    print(f"  Episodes: {episodes}")
    print(f"  Frames:   {total_frames:,}")
    print(f"  FPS:      {fps:,.0f}")

    state = env.reset()
    _check(state.shape == (4,), f"state.shape != (4,): {state.shape}")
    _check(state.dtype == np.float32, f"state.dtype != float32: {state.dtype}")
    print("  [OK] Env passed.")

    # ── 2. GANet ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("   GANet Smoke Test")
    print("=" * 50)

    ga = GANet()
    params = sum(p.numel() for p in ga.parameters())
    print(f"  Arch: 4 -> 16 -> 1, params: {params}")

    state = env.reset()
    action = ga.get_action(state)
    _check(action in (0, 1), f"action not in (0,1): {action}")
    sig = ga.get_sigmoid(state)
    _check(0.0 <= sig <= 1.0, f"sigmoid out of range: {sig}")
    print("  [OK] GANet passed.")

    # ── 3. GA Trainer ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("   GA Trainer (5 gens, pop=20)")
    print("=" * 50)

    result = train_ga(generations=5, population_size=20,
                      elitism=2, verbose=True)
    _check(isinstance(result['best_agent'], GANet), "best_agent not GANet")
    _check(len(result['history']) == 5, "history length != 5")
    print("  [OK] GA Trainer passed.")

    # ── 4. EML AST ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("   EML AST Smoke Test")
    print("=" * 50)

    basis = make_basis()
    tree = search.random_tree(basis, max_depth=3)
    print(f"  Random tree: {tree.to_string()}")
    print(f"  Depth: {tree.depth()}, Size: {tree.size()}")

    val = tree.evaluate(np.array([0.5, 0.0, 0.3, 0.5], dtype=np.float32))
    _check(not (val != val), f"NaN detected: {val}")
    print(f"  evaluate([0.5, 0, 0.3, 0.5]) = {val:.4f}")

    mutated = search.mutate_tree(tree, basis)
    _check(mutated.to_string() is not None, "mutated tree is None")
    print(f"  Mutated:     {mutated.to_string()}")

    _check(abs(eml_op(0.0, 1.0) - 1.0) < 1e-6, "eml(0,1) should be ~1")

    # Serialization round trip
    d = tree.to_dict()
    restored = EMLNode.from_dict(d)
    _check(restored.to_string() == tree.to_string(),
           "Serialisation roundtrip failed")
    print("  [OK] EML AST + serialisation passed.")

    # ── 5. Dataset Collection ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print("   Dataset Collection (10 episodes)")
    print("=" * 50)

    oracle = result['best_agent']
    states, targets_sig = collect_dataset(oracle, n_episodes=10,
                                          mode='sigmoid')
    _, targets_bin = collect_dataset(oracle, n_episodes=10, mode='binary')
    print(f"  Sigmoid: {len(states)} samples, "
          f"range [{targets_sig.min():.3f}, {targets_sig.max():.3f}]")
    print(f"  Binary:  unique values = {np.unique(targets_bin)}")
    _check(len(states) > 0, "empty dataset")
    print("  [OK] Dataset passed.")

    # ── 6. EML Evolution ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("   EML Evolution (20 gens, pop=50)")
    print("=" * 50)

    eml_result = evolve_eml(states, targets_sig, mode='sigmoid',
                            depth_penalty_name='medium',
                            generations=20, population_size=50,
                            patience=10, verbose=True)
    _check(eml_result['best_tree'] is not None, "no best tree")
    print(f"\n  Best formula: {eml_result['best_tree'].to_string()}")

    agent = EMLAgent(eml_result['best_tree'])
    action = agent.get_action(states[0])
    _check(action in (0, 1), f"action not in (0,1): {action}")
    print("  [OK] EML Evolution passed.")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  [OK] All tests passed!")
    print("=" * 50)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [gui|train|demo|test]")
        print("  gui    — graphical interface (PySide6)")
        print("  train  — full benchmark (GA + EML distillation)")
        print("  demo   — pygame demo")
        print("  test   — smoke test of the modules")
        return

    cmd = sys.argv[1].lower()
    if cmd == 'train':
        cmd_train()
    elif cmd == 'demo':
        cmd_demo()
    elif cmd == 'gui':
        cmd_gui()
    elif cmd == 'test':
        cmd_test()
    else:
        print(f"Unknown command: {cmd}")
        print("Use: gui, train, demo, test")


if __name__ == "__main__":
    main()

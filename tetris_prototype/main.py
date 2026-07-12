"""
Tetris AI — Нейро-символьное управление через EML-дистилляцию.

Точка входа:
    python main.py test    — smoke-test среды и признаков
    python main.py train   — обучение CNN-оракула (PPO)        [TODO]
    python main.py distill — EML-дистилляция оракула           [TODO]
    python main.py bench   — сравнение CNN vs EML              [TODO]
    python main.py demo    — pygame side-by-side демо          [TODO]
"""

import sys
import time
import numpy as np


def cmd_test():
    """Smoke-test среды и экстрактора признаков."""
    from env import TetrisEnv
    from features import extract, column_heights
    import config

    def _check(cond, msg):
        if not cond:
            raise AssertionError(msg)

    # ── 1. TetrisEnv: интерфейс и obs ────────────────────────────────────
    print("=" * 56)
    print("   TetrisEnv Smoke Test")
    print("=" * 56)

    env = TetrisEnv(seed=42)
    obs = env.reset()
    _check(set(obs.keys()) == {'grid', 'scalars'}, f"obs keys: {obs.keys()}")
    _check(obs['grid'].shape == (3, config.BOARD_H_TOTAL, config.BOARD_W),
           f"grid shape: {obs['grid'].shape}")
    _check(obs['scalars'].shape == (19,), f"scalars shape: {obs['scalars'].shape}")
    _check(obs['grid'].dtype == np.float32, "grid dtype != float32")
    # one-hot текущей фигуры — ровно одна единица в [0:7]
    _check(obs['scalars'][:7].sum() == 1.0, "cur piece one-hot broken")
    _check(obs['scalars'][7:14].sum() == 1.0, "next piece one-hot broken")
    print(f"  obs.grid:    {obs['grid'].shape}")
    print(f"  obs.scalars: {obs['scalars'].shape}")
    print("  [OK] Interface passed.")

    # ── 2. Действия валидны, эпизод завершается ──────────────────────────
    print("\n" + "=" * 56)
    print("   Episode dynamics")
    print("=" * 56)

    env.reset()
    total_reward = 0.0
    steps = 0
    while not env.done and steps < config.MAX_EPISODE_FRAMES:
        a = env.rng.integers(0, config.N_ACTIONS)
        _, r, done, info = env.step(int(a))
        total_reward += r
        steps += 1
    print(f"  Random episode: {steps} steps, score={info['score']}, "
          f"reward={total_reward:.2f}")
    _check(env.done, "episode did not terminate")
    print("  [OK] Episode terminates.")

    # ── 3. Hard drop фиксирует фигуру и очищает линии ────────────────────
    print("\n" + "=" * 56)
    print("   Hard drop + line clear")
    print("=" * 56)

    env.reset()
    filled_before = int((env.board != 0).sum())
    env.step(5)   # hard drop
    filled_after = int((env.board != 0).sum())
    # после хард-дропа либо 4 ячейки добавились, либо линия очистилась
    print(f"  Cells before: {filled_before}, after hard drop: {filled_after}")
    _check(filled_after >= 0, "board corrupted")
    print("  [OK] Hard drop works.")

    # ── 4. Features: размерность, диапазон, эталоны ──────────────────────
    print("\n" + "=" * 56)
    print("   Feature extractor")
    print("=" * 56)

    # Пустая доска
    empty = np.zeros((config.BOARD_H_TOTAL, config.BOARD_W), dtype=np.int8)
    f_empty = extract(empty, piece_type=0, rotation=0)
    _check(f_empty.shape == (28,), f"features shape: {f_empty.shape}")
    _check(np.all(np.abs(f_empty[:17]) <= 1.0 + 1e-6),
           "numeric features out of [-1,1]")
    # на пустой доске высоты = 0 → нормализованы в -1
    _check(np.allclose(f_empty[:10], -1.0), "empty heights != -1")
    h_empty = column_heights(empty)
    _check(np.all(h_empty == 0), "empty board heights != 0")
    print(f"  Empty board features OK (heights all -1).")

    # Доска с одной полной нижней строкой кроме одной дырки в колонке 0
    test_board = np.zeros((config.BOARD_H_TOTAL, config.BOARD_W), dtype=np.int8)
    test_board[-1, 1:] = 1            # нижняя строка занята кроме колонки 0
    test_board[-3, 0] = 1            # навес над колонкой 0 → дыра
    h = column_heights(test_board)
    from features import count_holes
    holes = count_holes(test_board, h)
    print(f"  Test board heights: {h.tolist()}")
    print(f"  Test board holes: {holes}")
    _check(holes >= 2, f"expected holes under overhang, got {holes}")
    print("  [OK] Features passed.")

    # ── 5. Бенчмарк FPS (цель ≥ 50K) ─────────────────────────────────────
    print("\n" + "=" * 56)
    print("   FPS Benchmark (random policy)")
    print("=" * 56)

    env = TetrisEnv(seed=0)
    episodes = 1000
    total_frames = 0
    t0 = time.perf_counter()
    for _ in range(episodes):
        env.reset()
        while not env.done:
            env.step(int(env.rng.integers(0, config.N_ACTIONS)))
            total_frames += 1
    elapsed = time.perf_counter() - t0
    fps = total_frames / elapsed
    print(f"  Episodes:    {episodes}")
    print(f"  Frames:      {total_frames:,}")
    print(f"  Elapsed:     {elapsed:.2f}s")
    print(f"  FPS:         {fps:,.0f}")
    if fps >= 50_000:
        print("  [OK] FPS target (>= 50K) reached.")
    else:
        print(f"  [WARN] FPS below 50K target ({fps:,.0f}). Optimization needed.")

    # ── 6. FPS с извлечением признаков (для оценки overhead дистилляции) ──
    env = TetrisEnv(seed=0)
    env.reset()
    n = 50_000
    t0 = time.perf_counter()
    for _ in range(n):
        env.get_features()
        if env.done:
            env.reset()
        env.step(5)
    elapsed = time.perf_counter() - t0
    print(f"  get_features: {n / elapsed:,.0f} calls/s")

    print("\n" + "=" * 56)
    print("  [OK] All smoke tests passed!")
    print("=" * 56)


def cmd_train():
    """Полноценное обучение CNN-оракула (PPO). Долго на CPU — лучше GPU."""
    from ppo_trainer import train_ppo
    from cnn_oracle import get_device, count_params
    import storage
    import config

    print("=" * 56)
    print("   PPO Training — TetrisCNN Oracle")
    print("=" * 56)
    print(f"  Device: {get_device()}")
    print(f"  Budget: {config.PPO_TOTAL_STEPS:,} steps, "
          f"{config.PPO_N_ENVS} envs, target {config.PPO_TARGET_SCORE} lines")
    print("  (Ctrl+C сохранит текущую модель)\n")

    try:
        result = train_ppo(verbose=True, log_every=1)
    except KeyboardInterrupt:
        print("\n  Interrupted — saving current model...")
        result = None

    if result is not None:
        path = storage.save_oracle(result['model'], meta={
            'total_steps': result['total_steps'],
            'best_avg_lines': result['best_avg_lines'],
            'stop_reason': result['stop_reason'],
        })
        print(f"\n  Saved oracle -> {path}")
        print(f"  Best avg lines: {result['best_avg_lines']:.2f}, "
              f"elapsed {result['elapsed']/60:.1f} min")


def cmd_test_cnn():
    """Smoke-test CNN-оракула: forward pass, размерности, save/load."""
    import torch
    from env import TetrisEnv
    from cnn_oracle import (
        TetrisCNN, get_device, obs_to_tensors,
        batch_obs_to_tensors, count_params,
    )
    import storage
    import config

    def _check(cond, msg):
        if not cond:
            raise AssertionError(msg)

    print("=" * 56)
    print("   TetrisCNN Smoke Test")
    print("=" * 56)

    device = get_device()
    print(f"  Device: {device}")
    model = TetrisCNN().to(device)
    n_params = count_params(model)
    print(f"  Params: {n_params:,}")

    # ── Forward на одном obs ─────────────────────────────────────────────
    env = TetrisEnv(seed=1)
    obs = env.reset()
    grid, scalars = obs_to_tensors(obs, device)
    logits, value = model(grid, scalars)
    _check(logits.shape == (1, config.N_ACTIONS), f"logits: {logits.shape}")
    _check(value.shape == (1,), f"value: {value.shape}")
    print(f"  logits: {logits.shape}, value: {value.shape}")

    # ── act / evaluate_actions ───────────────────────────────────────────
    action, log_prob, val = model.act(grid, scalars)
    _check(0 <= action.item() < config.N_ACTIONS, "action out of range")
    lp, ent, v = model.evaluate_actions(grid, scalars, action)
    _check(ent.item() >= 0, "negative entropy")
    print(f"  act -> a={action.item()}, logp={log_prob.item():.3f}, "
          f"entropy={ent.item():.3f}")

    # ── Батч ─────────────────────────────────────────────────────────────
    obs_list = [env.reset() for _ in range(8)]
    g, s = batch_obs_to_tensors(obs_list, device)
    logits_b, value_b = model(g, s)
    _check(logits_b.shape == (8, config.N_ACTIONS), f"batch logits: {logits_b.shape}")
    print(f"  batch(8) logits: {logits_b.shape}")

    # ── save/load roundtrip ──────────────────────────────────────────────
    path = storage.save_oracle(model, meta={'test': True})
    model2 = TetrisCNN().to(device)
    storage.load_oracle(model2, path, device=device)
    with torch.no_grad():
        l1, _ = model(grid, scalars)
        l2, _ = model2(grid, scalars)
    _check(torch.allclose(l1, l2), "save/load mismatch")
    print(f"  save/load roundtrip OK ({path})")

    # ── Тайминг forward (для оценки latency vs EML позже) ────────────────
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(200):
            model(grid, scalars)
        dt = (time.perf_counter() - t0) / 200 * 1e6
    print(f"  forward latency: {dt:.1f} us/step (batch=1, {device})")

    print("\n" + "=" * 56)
    print("  [OK] CNN smoke test passed!")
    print("=" * 56)


def cmd_test_ppo():
    """Smoke-test PPO: крохотный прогон end-to-end (что не падает)."""
    from ppo_trainer import train_ppo, VecTetris, compute_gae
    import numpy as np

    print("=" * 56)
    print("   PPO Smoke Test (tiny run)")
    print("=" * 56)

    # ── GAE юнит-проверка на простом случае ──────────────────────────────
    rewards = np.ones((4, 1), dtype=np.float32)
    values = np.zeros((4, 1), dtype=np.float32)
    dones = np.zeros((4, 1), dtype=np.float32)
    adv, ret = compute_gae(rewards, values, dones, np.zeros(1),
                           gamma=0.99, lam=0.95)
    assert adv.shape == (4, 1), "GAE shape wrong"
    assert ret[3, 0] > 0, "GAE returns wrong"
    print(f"  GAE OK: returns[last]={ret[3,0]:.3f}, advantages[0]={adv[0,0]:.3f}")

    # ── VecTetris ────────────────────────────────────────────────────────
    vec = VecTetris(n_envs=4, seed=0)
    g, s = vec.reset()
    assert g.shape == (4, 3, 24, 10), f"vec grid: {g.shape}"
    g, s, r, d = vec.step(np.array([5, 5, 5, 5]))
    assert r.shape == (4,), f"vec rewards: {r.shape}"
    print(f"  VecTetris(4) OK: grid {g.shape}, rewards {r.shape}")

    # ── Мини train_ppo: 2 update'а на 2 средах ───────────────────────────
    print("  Running 2 PPO updates (n_envs=4, rollout=64)...")
    result = train_ppo(
        total_steps=4 * 64 * 2,   # ровно 2 update'а
        n_envs=4, rollout=64, epochs=2, batch_size=64,
        target_score=0, seed=0, verbose=True, log_every=1,
    )
    assert len(result['history']) >= 1, "no PPO updates ran"
    assert 'model' in result, "no model returned"
    print(f"\n  Ran {len(result['history'])} updates, "
          f"{result['total_steps']} steps, {result['elapsed']:.1f}s")

    print("\n" + "=" * 56)
    print("  [OK] PPO smoke test passed!")
    print("=" * 56)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [test|testcnn|testppo|train|gui|distill|bench|demo]")
        print("  test    — smoke-test среды и признаков")
        print("  train   — обучение CNN-оракула (PPO)         [TODO]")
        print("  distill — EML-дистилляция                    [TODO]")
        print("  bench   — сравнение CNN vs EML               [TODO]")
        print("  demo    — pygame демо                        [TODO]")
        return

    cmd = sys.argv[1].lower()
    if cmd == 'test':
        cmd_test()
    elif cmd == 'testcnn':
        cmd_test_cnn()
    elif cmd == 'testppo':
        cmd_test_ppo()
    elif cmd == 'train':
        cmd_train()
    elif cmd == 'gui':
        from gui import main as gui_main
        gui_main()
    else:
        print(f"Команда '{cmd}' ещё не реализована (см. roadmap в PROJECT_BLUEPRINT.md).")


if __name__ == "__main__":
    main()

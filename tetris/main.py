"""
Tetris AI — neuro-symbolic control through EML distillation.

Entry point:
    python main.py test    — smoke test of the environment and the features
    python main.py train   — CNN oracle training (PPO, placement-based)
    python main.py distill — EML distillation of the oracle
    python main.py bench   — CNN vs EML comparison
    python main.py gui     — GUI (training + distillation)
"""

import sys
import time
import numpy as np


def cmd_test():
    """Smoke test of the environment and the feature extractor."""
    from env import TetrisEnv
    from features import extract_afterstate, column_heights, count_holes
    import config

    def _check(cond, msg):
        if not cond:
            raise AssertionError(msg)

    # ── 1. TetrisEnv: the interface and the placement obs ────────────────
    print("=" * 56)
    print("   TetrisEnv Smoke Test (placement mode)")
    print("=" * 56)

    env = TetrisEnv(seed=42)
    obs = env.reset()
    _check(set(obs.keys()) == {'grid', 'scalars', 'mask', 'afeats'},
           f"obs keys: {obs.keys()}")
    _check(obs['grid'].shape == (1, config.BOARD_H_TOTAL, config.BOARD_W),
           f"grid shape: {obs['grid'].shape}")
    _check(obs['scalars'].shape == (14,), f"scalars shape: {obs['scalars'].shape}")
    _check(obs['mask'].shape == (config.N_PLACEMENTS,),
           f"mask shape: {obs['mask'].shape}")
    _check(obs['afeats'].shape == (config.N_PLACEMENTS,
                                   config.N_AFTERSTATE_FEATURES),
           f"afeats shape: {obs['afeats'].shape}")
    _check(obs['grid'].dtype == np.float32, "grid dtype != float32")
    _check(obs['scalars'][:7].sum() == 1.0, "cur piece one-hot broken")
    _check(obs['scalars'][7:14].sum() == 1.0, "next piece one-hot broken")
    _check(obs['mask'].sum() >= 9, f"too few legal placements: {obs['mask'].sum()}")
    print(f"  obs.grid:    {obs['grid'].shape}")
    print(f"  obs.scalars: {obs['scalars'].shape}")
    print(f"  obs.mask:    {obs['mask'].shape}, legal={int(obs['mask'].sum())}")
    print(f"  obs.afeats:  {obs['afeats'].shape}")
    print("  [OK] Interface passed.")

    # ── 2. A placement episode with a random policy ──────────────────────
    print("\n" + "=" * 56)
    print("   Episode dynamics (random placements)")
    print("=" * 56)

    obs = env.reset()
    total_reward = 0.0
    steps = 0
    while not env.done and steps < config.MAX_EPISODE_PLACEMENTS:
        legal = np.flatnonzero(obs['mask'] > 0)
        _check(len(legal) > 0, "no legal placements but not done")
        a = int(env.rng.choice(legal))
        obs, r, done, info = env.step_placement(a)
        total_reward += r
        steps += 1
    print(f"  Random episode: {steps} placements, score={info['score']}, "
          f"reward={total_reward:.2f}")
    _check(env.done, "episode did not terminate")
    print("  [OK] Episode terminates.")

    # ── 3. afterstate == the actual placement ────────────────────────────
    print("\n" + "=" * 56)
    print("   Afterstate consistency")
    print("=" * 56)

    env = TetrisEnv(seed=7)
    obs = env.reset()
    legal = np.flatnonzero(obs['mask'] > 0)
    a = int(legal[len(legal) // 2])
    rot, xi = divmod(a, config.BOARD_W)
    board_pred, cleared_pred, landing = env.afterstate(rot, xi)
    env.step_placement(a)
    _check(np.array_equal(board_pred, env.board),
           "afterstate board != the actual board after step_placement")
    print(f"  placement a={a} (rot={rot}, xi={xi}), landing_h={landing}")
    print("  [OK] afterstate == step_placement.")

    # ── 4. Features: dimensionality, range, reference values ─────────────
    print("\n" + "=" * 56)
    print("   Feature extractor (afterstate)")
    print("=" * 56)

    empty = np.zeros((config.BOARD_H_TOTAL, config.BOARD_W), dtype=np.int8)
    f_empty = extract_afterstate(empty, cleared=0, landing_h=0)
    _check(f_empty.shape == (config.N_AFTERSTATE_FEATURES,),
           f"features shape: {f_empty.shape}")
    _check(np.all(np.abs(f_empty) <= 1.0 + 1e-6),
           "features out of [-1,1]")
    _check(np.allclose(f_empty[:10], -1.0), "empty heights != -1")

    test_board = np.zeros((config.BOARD_H_TOTAL, config.BOARD_W), dtype=np.int8)
    test_board[-1, 1:] = 1            # bottom row filled except column 0
    test_board[-3, 0] = 1             # an overhang above column 0 -> holes
    h = column_heights(test_board)
    holes = count_holes(test_board, h)
    print(f"  Test board heights: {h.tolist()}")
    print(f"  Test board holes: {holes}")
    _check(holes >= 2, f"expected holes under overhang, got {holes}")
    print("  [OK] Features passed.")

    # ── 5. Speed benchmark of placement steps ────────────────────────────
    print("\n" + "=" * 56)
    print("   Placement throughput (random policy)")
    print("=" * 56)

    env = TetrisEnv(seed=0)
    episodes = 300
    total_steps = 0
    t0 = time.perf_counter()
    for _ in range(episodes):
        obs = env.reset()
        while not env.done:
            legal = np.flatnonzero(obs['mask'] > 0)
            obs, _, _, _ = env.step_placement(int(env.rng.choice(legal)))
            total_steps += 1
    elapsed = time.perf_counter() - t0
    print(f"  Episodes:    {episodes}")
    print(f"  Placements:  {total_steps:,}")
    print(f"  Rate:        {total_steps / elapsed:,.0f} placements/s")

    # ── 6. Speed of afterstate_features (the EML inference load) ─────────
    env = TetrisEnv(seed=0)
    obs = env.reset()
    n = 0
    t0 = time.perf_counter()
    while n < 10_000:
        legal = np.flatnonzero(obs['mask'] > 0)
        for a in legal:
            rot, xi = divmod(int(a), config.BOARD_W)
            env.afterstate_features(rot, xi)
            n += 1
        obs, _, done, _ = env.step_placement(int(env.rng.choice(legal)))
        if done:
            obs = env.reset()
    elapsed = time.perf_counter() - t0
    print(f"  afterstate_features: {n / elapsed:,.0f} calls/s")

    print("\n" + "=" * 56)
    print("  [OK] All smoke tests passed!")
    print("=" * 56)


def cmd_train():
    """Full CNN oracle training (PPO)."""
    from ppo_trainer import train_ppo
    from cnn_oracle import describe_device
    import storage
    import reports
    import runs as runs_mod
    import config

    run_dir = runs_mod.create_run_dir('train')
    print("=" * 56)
    print("   PPO Training — TetrisCNN Oracle (placement-based)")
    print("=" * 56)
    print(f"  Device: {describe_device()}")
    print(f"  Run dir: {run_dir}")
    print(f"  Budget: {config.PPO_TOTAL_STEPS:,} placements, "
          f"{config.PPO_N_ENVS} envs, target {config.PPO_TARGET_SCORE} lines")
    print("  (Ctrl+C interrupts; the model is saved by the autopilot)\n")

    result = train_ppo(verbose=True, log_every=1, autopilot=True)

    meta = {
        'total_steps': result['total_steps'],
        'best_avg_lines': result['best_avg_lines'],
        'stop_reason': result['stop_reason'],
        'elapsed_min': round(result['elapsed'] / 60, 1),
    }
    path = storage.save_oracle(result['model'], meta=meta)
    print(f"\n  Saved oracle -> {path}")
    hist_path = reports.save_ppo_history(result['history'], meta=meta)
    png_path = reports.plot_ppo_history(result['history'])
    # A full snapshot of the run into the run directory.
    runs_mod.copy_into(run_dir, path, hist_path, png_path,
                       config.AUTOPILOT_SAVE_PATH)
    runs_mod.save_summary(run_dir, meta)
    print(f"  Reports -> {hist_path}")
    print(f"             {png_path}")
    print(f"  Run archive -> {run_dir}")
    print(f"  Best avg lines: {result['best_avg_lines']:.2f}, "
          f"elapsed {result['elapsed']/60:.1f} min")


def cmd_test_cnn():
    """Smoke test of the CNN oracle: forward pass, shapes, save/load."""
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

    # ── Forward on a single obs ──────────────────────────────────────────
    env = TetrisEnv(seed=1)
    obs = env.reset()
    grid, scalars, afeats, mask = obs_to_tensors(obs, device)
    logits, value = model(grid, scalars, afeats, mask)
    _check(logits.shape == (1, config.N_PLACEMENTS), f"logits: {logits.shape}")
    _check(value.shape == (1,), f"value: {value.shape}")
    # illegal placements must be masked out
    illegal = mask[0] <= 0
    _check(bool((logits[0][illegal] < -1e8).all()), "mask not applied")
    print(f"  logits: {logits.shape}, value: {value.shape}, mask applied OK")

    # ── act / evaluate_actions ───────────────────────────────────────────
    action, log_prob, val = model.act(grid, scalars, afeats, mask)
    _check(0 <= action.item() < config.N_PLACEMENTS, "action out of range")
    _check(bool(mask[0][action.item()] > 0), "sampled illegal action")
    lp, ent, v = model.evaluate_actions(grid, scalars, afeats, mask, action)
    _check(ent.item() >= 0, "negative entropy")
    print(f"  act -> a={action.item()}, logp={log_prob.item():.3f}, "
          f"entropy={ent.item():.3f}")

    # ── Batch ────────────────────────────────────────────────────────────
    obs_list = [env.reset() for _ in range(8)]
    g, s, af, m = batch_obs_to_tensors(obs_list, device)
    logits_b, value_b = model(g, s, af, m)
    _check(logits_b.shape == (8, config.N_PLACEMENTS),
           f"batch logits: {logits_b.shape}")
    print(f"  batch(8) logits: {logits_b.shape}")

    # ── save/load roundtrip ──────────────────────────────────────────────
    import os, tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), 'cnn_smoke_test.pt')
    path = storage.save_oracle(model, path=tmp_path, meta={'test': True})
    model2 = TetrisCNN().to(device)
    storage.load_oracle(model2, path, device=device)
    with torch.no_grad():
        l1, _ = model(grid, scalars, afeats, mask)
        l2, _ = model2(grid, scalars, afeats, mask)
    _check(torch.allclose(l1, l2), "save/load mismatch")
    print(f"  save/load roundtrip OK ({path})")

    # ── Forward timing (for the latency comparison against EML later) ────
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(200):
            model(grid, scalars, afeats, mask)
        dt = (time.perf_counter() - t0) / 200 * 1e6
    print(f"  forward latency: {dt:.1f} us/step (batch=1, {device})")

    print("\n" + "=" * 56)
    print("  [OK] CNN smoke test passed!")
    print("=" * 56)


def cmd_test_ppo():
    """Smoke test of PPO: a tiny end-to-end run (that it does not crash)."""
    from ppo_trainer import train_ppo, VecTetris, compute_gae
    import numpy as np

    print("=" * 56)
    print("   PPO Smoke Test (tiny run)")
    print("=" * 56)

    # ── GAE unit check on a simple case ──────────────────────────────────
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
    g, s, af, m = vec.reset()
    assert g.shape == (4, 1, 24, 10), f"vec grid: {g.shape}"
    assert af.shape == (4, 40, 19), f"vec afeats: {af.shape}"
    assert m.shape == (4, 40), f"vec mask: {m.shape}"
    # a step with legal actions
    acts = np.array([int(np.flatnonzero(m[i] > 0)[0]) for i in range(4)])
    g, s, af, m, r, d = vec.step(acts)
    assert r.shape == (4,), f"vec rewards: {r.shape}"
    print(f"  VecTetris(4) OK: grid {g.shape}, afeats {af.shape}, "
          f"rewards {r.shape}")

    # ── Mini train_ppo: 2 updates over 4 environments ────────────────────
    print("  Running 2 PPO updates (n_envs=4, rollout=32)...")
    result = train_ppo(
        total_steps=4 * 32 * 2,   # exactly 2 updates
        n_envs=4, rollout=32, epochs=2, batch_size=64,
        target_score=0, seed=0, verbose=True, log_every=1,
    )
    assert len(result['history']) >= 1, "no PPO updates ran"
    assert 'model' in result, "no model returned"
    print(f"\n  Ran {len(result['history'])} updates, "
          f"{result['total_steps']} steps, {result['elapsed']:.1f}s")

    print("\n" + "=" * 56)
    print("  [OK] PPO smoke test passed!")
    print("=" * 56)


def cmd_test_distill():
    """Smoke test of distillation: a tiny end-to-end run (that it does not crash)."""
    import os
    import tempfile
    import numpy as np
    from cnn_oracle import TetrisCNN, get_device
    from dataset_collector import collect_dataset, compute_sample_weights
    from eml_distiller import (
        EMLNode, EMLPolicy, random_tree, distill, play_episodes,
    )
    from env import TetrisEnv
    import storage
    import config

    def _check(cond, msg):
        if not cond:
            raise AssertionError(msg)

    print("=" * 56)
    print("   EML Distillation Smoke Test")
    print("=" * 56)

    device = get_device()

    # ── AST round-trip + vectorized vs scalar eval ───────────────────────
    tree = random_tree(max_depth=4, force_eml=True)
    d = tree.to_dict()
    tree2 = EMLNode.from_dict(d)
    X = np.random.uniform(-1, 1, size=(64, config.N_FEATURES)).astype(np.float64)
    batch = tree.evaluate_batch(X)
    scalar = np.array([tree2.evaluate(X[i]) for i in range(len(X))])
    _check(batch.shape == (64,), "evaluate_batch shape")
    _check(np.allclose(batch, scalar, atol=1e-6), "batch vs scalar eval mismatch")
    _check(np.all(np.isfinite(batch)), "non-finite eval")
    print("  [OK] AST eval (batch == scalar), serialization round-trip.")

    # ── EMLPolicy + storage round-trip ───────────────────────────────────
    pol = EMLPolicy(tree)
    env = TetrisEnv(seed=3)
    env.reset()
    a = pol.get_action(env)
    _check(0 <= a < config.N_PLACEMENTS, "policy action out of range")
    _check(env.placement_mask()[a] > 0, "policy chose illegal placement")
    tmp_path = os.path.join(tempfile.gettempdir(), 'eml_smoke_test.json')
    path = storage.save_formulas([tree], path=tmp_path, meta={'test': True})
    loaded, _ = storage.load_formulas(EMLNode, path)
    _check(len(loaded) == 1, "load_formulas count")
    print(f"  [OK] EMLPolicy + save/load round-trip ({path}).")

    # ── Tiny end-to-end distill ──────────────────────────────────────────
    oracle = TetrisCNN().to(device)
    try:
        storage.load_oracle(oracle, device=device)
        print("  Loaded trained oracle.")
    except (FileNotFoundError, RuntimeError, KeyError):
        print("  No compatible oracle — using random-init (smoke only).")

    print("  Collecting tiny dataset (3 episodes)...")
    feats, logits, groups = collect_dataset(
        oracle, n_episodes=3, device=device,
        max_placements=40, verbose=False)
    _check(feats.shape[1] == config.N_FEATURES, "feature dim")
    _check(logits.ndim == 1, "logit dim")
    w = compute_sample_weights(logits, groups)
    _check(abs(w.mean() - 1.0) < 1e-6, "weights not mean-normalized")
    print(f"  dataset: {feats.shape}, groups: {len(np.unique(groups))}")

    print("  Running tiny distill (data_gen=4, pop=20, game small)...")
    config_backup = (config.EML_JOINT_GENERATIONS, config.EML_JOINT_POPULATION,
                     config.EML_INGAME_GAMES, config.EML_INGAME_MAX_PLACEMENTS)
    config.EML_JOINT_GENERATIONS = 2
    config.EML_JOINT_POPULATION = 6
    config.EML_INGAME_GAMES = 1
    config.EML_INGAME_MAX_PLACEMENTS = 40
    try:
        res = distill(
            feats, logits, w,
            data_generations=4, data_population=20,
            joint=True, verbose=False,
        )
    finally:
        (config.EML_JOINT_GENERATIONS, config.EML_JOINT_POPULATION,
         config.EML_INGAME_GAMES, config.EML_INGAME_MAX_PLACEMENTS) = config_backup
    _check(len(res['trees']) == 1, "distill trees count")
    lines, steps = play_episodes(
        res['trees'][0], n_games=2, max_placements=40, seed=7)
    print(f"  distilled formula: {lines:.1f} lines, {steps:.0f} placements/game")

    print("\n" + "=" * 56)
    print("  [OK] Distillation smoke test passed!")
    print("=" * 56)


def cmd_distill():
    """EML distillation: oracle -> formula / batch of formulas."""
    from cnn_oracle import get_device
    from pipeline import full_distill, batch_distill
    import config

    # Flags: --reuse-data (dataset cache), --no-joint (without the GAME phase),
    #        --batch [N] (a batch of formulas), --spread X (parameter spread).
    args = [a.lower() for a in sys.argv[2:]]
    reuse_data = '--reuse-data' in args
    do_joint = '--no-joint' not in args

    batch_n = None
    if '--batch' in args:
        batch_n = config.EML_BATCH_VARIANTS
        idx = args.index('--batch')
        if idx + 1 < len(args):
            try:
                batch_n = int(args[idx + 1])
            except ValueError:
                pass
    spread = config.EML_BATCH_SPREAD
    if '--spread' in args:
        idx = args.index('--spread')
        if idx + 1 < len(args):
            try:
                spread = float(args[idx + 1])
            except ValueError:
                pass

    print("=" * 56)
    print("   EML Distillation — TetrisCNN -> formula")
    print("=" * 56)
    mode = f"batch x{batch_n} (spread ±{spread:.0%})" if batch_n else "single"
    print(f"  Device: {get_device()}  | {mode}  reuse_data={reuse_data}  "
          f"joint={do_joint}")

    if batch_n:
        summary = batch_distill(
            n_variants=batch_n, spread=spread,
            reuse_data=reuse_data, joint=do_joint, verbose=True)
    else:
        summary = full_distill(
            reuse_data=reuse_data, joint=do_joint, verbose=True)

    print("\n" + "=" * 56)
    print("   Distillation summary")
    print("=" * 56)
    if 'variants' in summary:
        print(f"  {'#':<3}{'lines':>8}{'size':>6}{'depth':>7}"
              f"{'dp':>8}{'gen':>6}{'pop':>6}")
        for v in sorted(summary['variants'],
                        key=lambda x: -x['final_lines']):
            p = v['params']
            print(f"  {v['variant']:<3}{v['final_lines']:>8.1f}"
                  f"{v['size']:>6}{v['depth']:>7}{p['depth_penalty']:>8}"
                  f"{p['generations']:>6}{p['population']:>6}")
        print(f"  Run dir: {summary['run_dir']}")
    t = summary['trees'][0]
    print(f"\n  BEST f[PLACE]  D{t.depth()} S{t.size()} V{t.n_unique_vars()}:")
    print(f"    {t.to_string()[:200]}")
    print(f"\n  EML lines/game:    {summary['eml_lines']:.2f}")
    print(f"  Oracle lines/game: {summary['oracle_lines']:.2f}")
    print(f"  EML / Oracle:      {summary['ratio_pct']:.1f}%  (target >= 50%)")
    print(f"  AST size:          {summary['total_size']} nodes")
    print(f"  Saved formula ->   {summary['path']}")


def cmd_bench():
    """CNN oracle vs EML formula comparison on identical seeds."""
    import time
    import numpy as np
    import torch
    from env import TetrisEnv
    from cnn_oracle import TetrisCNN, get_device, obs_to_tensors
    from eml_distiller import EMLNode, EMLPolicy
    import storage
    import config

    n_games = 30
    if len(sys.argv) > 2:
        try:
            n_games = int(sys.argv[2])
        except ValueError:
            pass

    print("=" * 56)
    print(f"   Benchmark: CNN vs EML  ({n_games} games, same seeds)")
    print("=" * 56)

    device = get_device()
    oracle = TetrisCNN().to(device)
    storage.load_oracle(oracle, device=device)
    oracle.eval()
    trees, meta = storage.load_formulas(EMLNode)
    policy = EMLPolicy(trees)
    print(f"  EML formula meta: {meta}")

    max_steps = config.MAX_EPISODE_PLACEMENTS
    orc_lines, eml_lines = [], []
    orc_steps, eml_steps = [], []
    orc_lat, eml_lat = [], []

    for g in range(n_games):
        # Oracle.
        env = TetrisEnv(seed=g)
        obs = env.reset()
        f = 0
        while not env.done and f < max_steps:
            grid, scalars, afeats, mask = obs_to_tensors(obs, device)
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = oracle.get_logits(grid, scalars, afeats, mask)
            a = int(torch.argmax(logits, dim=-1).item())
            orc_lat.append((time.perf_counter() - t0) * 1e6)
            obs, _, _, _ = env.step_placement(a)
            f += 1
        orc_lines.append(env.score)
        orc_steps.append(f)

        # EML.
        env = TetrisEnv(seed=g)
        obs = env.reset()
        f = 0
        while not env.done and f < max_steps:
            t0 = time.perf_counter()
            a = policy.choose(obs['mask'], obs['afeats'], obs['scalars'])
            eml_lat.append((time.perf_counter() - t0) * 1e6)
            if a < 0:
                break
            obs, _, _, _ = env.step_placement(a)
            f += 1
        eml_lines.append(env.score)
        eml_steps.append(f)

    def _stats(xs):
        arr = np.asarray(xs, dtype=np.float64)
        return arr.mean(), arr.max(), np.median(arr)

    ol_avg, ol_max, _ = _stats(orc_lines)
    el_avg, el_max, _ = _stats(eml_lines)
    ratio = (el_avg / ol_avg * 100.0) if ol_avg > 0 else 0.0
    o_lat = float(np.median(orc_lat))
    e_lat = float(np.median(eml_lat))
    speedup = (o_lat / e_lat) if e_lat > 0 else 0.0

    print(f"\n  {'metric':<22}{'CNN':>12}{'EML':>12}")
    print("  " + "-" * 46)
    print(f"  {'avg lines':<22}{ol_avg:>12.2f}{el_avg:>12.2f}")
    print(f"  {'max lines':<22}{ol_max:>12.0f}{el_max:>12.0f}")
    print(f"  {'avg placements':<22}{np.mean(orc_steps):>12.0f}"
          f"{np.mean(eml_steps):>12.0f}")
    print(f"  {'latency us/decision':<22}{o_lat:>12.2f}{e_lat:>12.2f}")
    print("  " + "-" * 46)
    print(f"  EML / CNN score:   {ratio:.1f}%   (target >= 50%)")
    print(f"  EML speedup:       {speedup:.1f}x")
    total_size = sum(t.size() for t in trees)
    print(f"  AST size:          {total_size} nodes (target < 50)")

    # ── CSV output ───────────────────────────────────────────────────────
    import os, csv
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(config.RESULTS_DIR, 'benchmark_table.csv')
    with open(table_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'cnn', 'eml'])
        w.writerow(['avg_lines', f"{ol_avg:.2f}", f"{el_avg:.2f}"])
        w.writerow(['max_lines', f"{ol_max:.0f}", f"{el_max:.0f}"])
        w.writerow(['avg_placements', f"{np.mean(orc_steps):.0f}",
                    f"{np.mean(eml_steps):.0f}"])
        w.writerow(['latency_us_median', f"{o_lat:.2f}", f"{e_lat:.2f}"])
        w.writerow(['eml_vs_cnn_pct', '', f"{ratio:.1f}"])
        w.writerow(['eml_speedup_x', '', f"{speedup:.1f}"])
        w.writerow(['ast_size_nodes', '', str(total_size)])
        w.writerow(['n_games', str(n_games), str(n_games)])
    games_path = os.path.join(config.RESULTS_DIR, 'benchmark_games.csv')
    with open(games_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['seed', 'cnn_lines', 'cnn_placements',
                    'eml_lines', 'eml_placements'])
        for g in range(n_games):
            w.writerow([g, orc_lines[g], orc_steps[g],
                        eml_lines[g], eml_steps[g]])
    print(f"  CSV -> {table_path}")
    print(f"         {games_path}")


def cmd_export():
    """Export the EML formula to inline C (for the MCU / latency measurement)."""
    import os
    from eml_distiller import EMLNode
    import storage
    import config

    src = sys.argv[2] if len(sys.argv) > 2 else None
    trees, meta = storage.load_formulas(EMLNode, src)
    tree = trees[0]

    def _c_expr(node) -> str:
        if node.kind == 'const':
            return f"{node.value:.6f}f"
        if node.kind == 'var':
            return f"f[{node.var_idx}]"
        return f"eml_op({_c_expr(node.left)}, {_c_expr(node.right)})"

    lines = [
        "/*",
        " * eml_formula.h — distilled Tetris EML policy (auto-generated).",
        " *",
        f" * Source: {src or 'models/best_eml.json'}",
        f" * AST: size={tree.size()}  depth={tree.depth()}  "
        f"unique_vars={tree.n_unique_vars()}",
        f" * Distillation metrics: {meta}",
        " *",
        " * Usage: for every legal placement fill f[26] (normalized as in",
        " * features.py: [-1,1]) and choose the placement with the maximum",
        " * eml_formula(f).",
        " *",
        " * Feature indices:",
    ]
    for i, name in enumerate(config.FEATURE_NAMES):
        lines.append(f" *   f[{i:2d}] = {name}")
    lines += [
        " */",
        "#ifndef EML_FORMULA_H",
        "#define EML_FORMULA_H",
        "",
        "#include <math.h>",
        "",
        f"#define EML_EPS {config.EML_EPSILON:g}f",
        "",
        "static inline float eml_op(float x, float y) {",
        "    if (x > 10.0f) x = 10.0f;",
        "    if (x < -10.0f) x = -10.0f;",
        "    return expf(x) - logf(fabsf(y) + EML_EPS);",
        "}",
        "",
        "static inline float eml_formula(const float f[26]) {",
        f"    return {_c_expr(tree)};",
        "}",
        "",
        "#endif /* EML_FORMULA_H */",
        "",
    ]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, 'eml_formula.h')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))

    print("=" * 56)
    print("   EML -> C export")
    print("=" * 56)
    print(f"  AST size {tree.size()}, depth {tree.depth()}, "
          f"vars {tree.n_unique_vars()}")
    print(f"  Saved -> {path}")
    print("  The formula is a single C expression of expf/logf: it evaluates on")
    print("  an MCU in a few microseconds; a CNN oracle does not fit there.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py "
              "[test|testcnn|testppo|testdistill|train|gui|distill|bench]")
        print("  test        — smoke test of the environment and the features")
        print("  testcnn     — smoke test of the CNN oracle")
        print("  testppo     — smoke test of PPO")
        print("  testdistill — smoke test of EML distillation")
        print("  train       — CNN oracle training (PPO, placement-based)")
        print("  distill     — EML distillation of the oracle into a formula")
        print("                 [--reuse-data] [--no-joint]")
        print("  bench [N]   — CNN vs EML over N games (+CSV in results/)")
        print("  export [f]  — export the EML formula to inline C "
              "(results/eml_formula.h)")
        print("  gui         — GUI (training · distillation · play · formulas)")
        print("  play        — GUI on the Play tab (watch the model play)")
        return

    cmd = sys.argv[1].lower()
    if cmd == 'test':
        cmd_test()
    elif cmd == 'testcnn':
        cmd_test_cnn()
    elif cmd == 'testppo':
        cmd_test_ppo()
    elif cmd == 'testdistill':
        cmd_test_distill()
    elif cmd == 'train':
        cmd_train()
    elif cmd == 'distill':
        cmd_distill()
    elif cmd == 'bench':
        cmd_bench()
    elif cmd == 'gui':
        from gui import main as gui_main
        gui_main()
    elif cmd == 'play':
        from gui import main as gui_main
        gui_main(initial_tab='play')
    elif cmd == 'export':
        cmd_export()
    else:
        print(f"Command '{cmd}' not found (see python main.py).")


if __name__ == "__main__":
    main()

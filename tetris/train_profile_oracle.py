"""
Train the PPO oracle on the current difficulty profile (EML_TETRIS_ENV).

    EML_TETRIS_ENV=garbage python train_profile_oracle.py --out models/garbage_ppo.pt

Same trainer and hyperparameters as `main.py train`, with the early stop on a
line target turned off (the garbage profile has its own scale) and a fixed
budget instead. The checkpoint with the best rolling average is what the
autopilot keeps; the final weights are saved to --out with a meta file.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config                      # noqa: E402
import storage                     # noqa: E402
from ppo_trainer import train_ppo  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=2_000_000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    t0 = time.time()
    res = train_ppo(total_steps=a.steps, target_score=0, seed=a.seed,
                    verbose=True, log_every=10, autopilot=True)
    meta = {'profile': config.TETRIS_ENV,
            'profile_params': config.TETRIS_PROFILES[config.TETRIS_ENV],
            'total_steps': res['total_steps'],
            'best_avg_lines': res['best_avg_lines'],
            'stop_reason': res['stop_reason'], 'seed': a.seed,
            'elapsed_min': round((time.time() - t0) / 60, 1)}
    storage.save_oracle(res['model'], path=a.out, meta=meta)
    with open(os.path.splitext(a.out)[0] + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print('saved', a.out, meta)


if __name__ == '__main__':
    main()

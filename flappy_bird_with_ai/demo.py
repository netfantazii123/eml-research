"""
Pygame демо — все агенты играют одновременно.
GA (красный), EML-weak (жёлтый), EML-medium (зелёный), EML-strong (синий).

Может принимать:
    - dict {name: agent} (как раньше)
    - список путей к .pt / .json (см. load_agents_from_paths)
    - None — автозагрузит последнюю GA модель из models/
"""

import os
import numpy as np
from config import (
    SCREEN_W, SCREEN_H, PIPE_GAP, PIPE_WIDTH, BIRD_X, BIRD_RADIUS,
)
from env import FlappyEnv
from storage import (
    list_saved_runs, load_eml_formula, latest_ga_path,
    load_ga_model as _load_ga,
)


COLORS = {
    'GA': (231, 76, 60),
    'EML-weak': (241, 196, 15),
    'eml-weak': (241, 196, 15),
    'EML-medium': (46, 204, 113),
    'eml-medium': (46, 204, 113),
    'EML-strong': (52, 152, 219),
    'eml-strong': (52, 152, 219),
}


def _color_for(name: str) -> tuple[int, int, int]:
    if name in COLORS:
        return COLORS[name]
    for key, c in COLORS.items():
        if key.lower() in name.lower():
            return c
    return (200, 200, 200)


def load_agents_from_paths(paths: list[str]) -> dict:
    """
    Загрузить агенты из списка путей. Имя в dict — basename без расширения.
    Распознаёт .pt (GA) и .json (EML).
    """
    agents = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"  Skip (not found): {path}")
            continue
        ext = os.path.splitext(path)[1].lower()
        name = os.path.splitext(os.path.basename(path))[0]
        if ext == '.pt':
            agents[name] = _load_ga(path)
        elif ext == '.json':
            data = load_eml_formula(path)
            agents[name] = data['agent']
        else:
            print(f"  Skip (unknown ext): {path}")
    return agents


def load_latest_per_method() -> dict:
    """Последняя сохранённая модель каждого метода (GA + 3 EML)."""
    by_method: dict[str, str] = {}
    for r in list_saved_runs():
        m = r['method']
        if m not in by_method:
            by_method[m] = r['path']
    return load_agents_from_paths(list(by_method.values()))


def run_demo(agents: dict | None = None, fps: int = 60):
    """
    Запустить pygame-демо: все агенты играют одновременно.

    Args:
        agents: dict {name: agent}. Если None — последняя GA модель.
        fps: кадров в секунду.
    """
    import pygame

    if agents is None:
        path = latest_ga_path()
        if path is None:
            print("  No models found. Run benchmark first.")
            return
        agents = {'GA': _load_ga(path)}

    if not agents:
        print("  No agents provided.")
        return

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Flappy Bird AI Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18, bold=True)

    BG_COLOR = (30, 30, 46)
    PIPE_COLOR = (69, 123, 78)

    def reset_all():
        # Изолируем глобальный RNG от наших манипуляций с сидом.
        rng_state = np.random.get_state()
        seed = int(np.random.randint(0, 100000))
        envs = {}
        for name in agents:
            np.random.seed(seed)
            envs[name] = FlappyEnv()
        np.random.set_state(rng_state)
        alive = {name: True for name in agents}
        scores = {name: 0 for name in agents}
        return envs, alive, scores

    envs, alive, scores = reset_all()
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    envs, alive, scores = reset_all()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if paused:
            clock.tick(fps)
            continue

        all_dead = True
        for name, agent in agents.items():
            if not alive[name]:
                continue
            all_dead = False
            env = envs[name]
            state = env.get_state()
            action = agent.get_action(state)
            env.step(action)
            scores[name] = env.score
            if env.done:
                alive[name] = False

        if all_dead:
            pygame.time.wait(1500)
            envs, alive, scores = reset_all()

        # ── Отрисовка ────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        ref_env = None
        for name in agents:
            if alive[name]:
                ref_env = envs[name]
                break
        if ref_env is None:
            ref_env = list(envs.values())[0]

        for pipe in ref_env.pipes:
            px = int(pipe['x'])
            gap_y = int(pipe['gap_y'])
            top_h = gap_y - PIPE_GAP // 2
            bottom_y = gap_y + PIPE_GAP // 2
            pygame.draw.rect(screen, PIPE_COLOR,
                             (px, 0, PIPE_WIDTH, top_h))
            pygame.draw.rect(screen, PIPE_COLOR,
                             (px, bottom_y, PIPE_WIDTH, SCREEN_H - bottom_y))

        for name, env in envs.items():
            color = _color_for(name)
            alpha = 255 if alive[name] else 80
            bird_color = tuple(c * alpha // 255 for c in color)
            pygame.draw.circle(screen, bird_color,
                               (BIRD_X, int(env.bird_y)), BIRD_RADIUS)
            ex = BIRD_X + 4
            ey = int(env.bird_y) - 3
            pygame.draw.circle(screen, (255, 255, 255), (ex, ey), 3)
            pygame.draw.circle(screen, (0, 0, 0), (ex + 1, ey), 1)

        y_offset = 10
        for name in agents:
            color = _color_for(name)
            status = "DEAD" if not alive[name] else f"{scores[name]}"
            label = font.render(f"{name}: {status}", True, color)
            screen.blit(label, (10, y_offset))
            y_offset += 24

        hint = font.render("R=restart  SPACE=pause  ESC=exit",
                           True, (120, 120, 140))
        screen.blit(hint, (SCREEN_W - hint.get_width() - 10, SCREEN_H - 28))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()

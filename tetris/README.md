# Tetris + EML-дистилляция (этап 2, текущий)

Масштабирование EML-пайплайна: оракул — CNN, обученная PPO с placement-based действиями (поворот × колонка, 40 действий с маской легальности). Политика дистиллируется в одну EML-формулу над инженерными признаками доски (afterstate-оценка в духе Dellacherie).

**Результаты (июль 2026):** лучшая формула играет **355.6 линий/игру = 97.4 % от оракула** при 43 узлах AST и латентности в 5 раз ниже CNN (103 vs 518 мкс). Цели диплома (≥50 % от оракула, <50 узлов) достигнуты.

## Запуск

```bash
pip install -r requirements.txt
python main.py            # список команд
python main.py train      # обучение CNN-оракула (PPO), ~90 мин на CPU
python main.py distill    # EML-дистилляция оракула в формулу
python main.py bench 30   # сравнение CNN vs EML, CSV в results/
python main.py export     # экспорт формулы в C-инлайн (results/eml_formula.h)
python main.py gui        # GUI: обучение · дистилляция · play · формулы
```

Итоговые формулы — `models/best_eml*.json`, готовый C-заголовок для микроконтроллера — `results/eml_formula.h`, таблицы для диплома — `results/benchmark_table.csv`. Чекпоинты оракула (`*.pt`) в репозиторий не входят — переобучаются командой `train`.

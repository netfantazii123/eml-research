# Flappy Bird + EML-дистилляция (этап 1, завершён)

Proof-of-concept пайплайна **NN-оракул → датасет → GA-эволюция EML-формулы** на простой 2D-задаче. Оракул — крошечный MLP, обученный генетическим алгоритмом; его политика дистиллируется в EML-формулы трёх уровней сложности (weak / medium / strong).

Концепт EML-дистилляции целиком (базовый оператор, правила стабильности, роль этапов) описан в [`update.md`](update.md).

## Запуск

```bash
pip install -r requirements.txt
python main.py        # список команд
python main.py train  # полный бенчмарк: GA-обучение + EML-дистилляция
python main.py gui    # GUI (PySide6)
python main.py demo   # pygame-демо обученного агента
```

Готовые EML-формулы лежат в `models/*.json`, графики бенчмарков — в `results/`.

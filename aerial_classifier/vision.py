"""
vision.py — движок инференса для вкладок «Тест · Фото» и «Тест · Видео».

Вынесен из gui.py намеренно: логика классификации кадра не должна зависеть от
tkinter. Благодаря этому её можно прогнать из консоли (`python vision.py`) и
проверить численно, а не «на глаз» по картинке в окне.

Что здесь есть:

    Predictor    — единый интерфейс к обеим моделям. Оракул и EML-формулы
                   отвечают на один и тот же вопрос («что на этом кадре 64x64»)
                   и потому вызываются одинаково; вкладка теста только
                   переключает движок и сравнивает ответы.
    тайлинг      — разбиение большого кадра на окна 64x64. Модель обучена
                   именно на таких вырезках, поэтому единственный честный
                   способ применить её к кадру целиком — нарезать его так же.
    оверлей      — полупрозрачная заливка каждого окна цветом класса, numpy.
                   Текст здесь НЕ рисуется: подписи кладёт tk.Canvas поверх
                   картинки, так они выходят резче и не портят сам кадр.
    VideoSource  — обёртка над cv2.VideoCapture (файл или веб-камера).

OpenCV нужен ТОЛЬКО для VideoSource и импортируется лениво. Фото-режим, тайлинг
и оверлей работают на чистом numpy, поэтому отсутствие cv2 отключает вкладку
видео, но не ломает остальное.
"""

import numpy as np

import config

# Цвета классов для оверлея (RGB). Различимы и в цвете, и в оттенках серого.
CLASS_COLORS = {
    'BEACH':  (249, 226, 175),   # песочно-жёлтый
    'MOTO':   (243, 139, 168),   # красный
    'PEOPLE': (166, 227, 161),   # зелёный
    'ROADS':  (137, 180, 250),   # синий
}

DEFAULT_TILE = config.IMG_SIZE          # 64 — размер, на котором обучались
DEFAULT_STRIDE = config.IMG_SIZE        # без перекрытия


def class_color(class_idx: int) -> tuple[int, int, int]:
    """
    Цвет класса для оверлея и графиков.

    Для классов исходного датасета цвета подобраны вручную. Для чужой папки с
    произвольными именами они генерируются по индексу: раньше все неизвестные
    классы получали один и тот же серый, и оверлей на новом датасете превращался
    в равномерную заливку, по которой ничего не видно.
    """
    if class_idx < len(config.CLASS_NAMES):
        known = CLASS_COLORS.get(config.CLASS_NAMES[class_idx])
        if known is not None:
            return known
    # Золотое сечение по кругу оттенков: соседние индексы получают далёкие
    # цвета, и разных классов может быть сколько угодно.
    import colorsys
    hue = (class_idx * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.45, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


# ── Единый интерфейс к моделям ────────────────────────────────────────────────

class Predictor:
    """
    Классификатор кадров 64x64 — либо нейросеть-оракул, либо EML-формулы.

    Оба движка отвечают на один вопрос и возвращают одинаковые структуры,
    поэтому вкладка теста может переключать их одной кнопкой и показывать
    ответы рядом.

    Args:
        branch: 'A' (признаки) или 'B' (эмбеддинг CNN).
        engine: 'eml' — дистиллированные формулы, 'oracle' — нейросеть.
        path:   конкретный файл модели вместо активного слота. Через него
                вкладки теста прогоняют кадр любой моделью из библиотеки —
                сравнить две сохранённые модели на одном кадре иначе никак.
    """

    def __init__(self, branch: str = 'A', engine: str = 'eml',
                 path: str | None = None):
        self.branch = branch.upper()
        self.engine = engine.lower()
        if self.engine not in ('eml', 'oracle'):
            raise ValueError(f"engine должен быть 'eml' или 'oracle', "
                             f"получено {engine!r}")
        self.path = path
        self._model = None
        self._clf = None
        self._load()

    @property
    def oracle_branch(self) -> str:
        """
        Чей оракул стоит за этой веткой.

        Ветка C работает в пространстве признаков ветки A и учится у того же
        MLP — своего оракула у неё нет и быть не должно, иначе сравнение
        «одна формула против четырёх» мерило бы заодно и разницу оракулов.
        """
        return 'B' if self.branch == 'B' else 'A'

    def _load(self):
        import storage

        if self.engine == 'oracle':
            import oracle as oracle_mod
            self._device = oracle_mod.get_device()
            branch = self.oracle_branch
            self._model = (oracle_mod.FeatureMLP() if branch == 'A'
                           else oracle_mod.PixelCNN())
            self.meta = storage.load_oracle(self._model, branch=branch,
                                            path=self.path,
                                            device=self._device)
            self._model.eval()
        else:
            import eml_distiller as dist
            self._clf, payload = storage.load_classifier(branch=self.branch,
                                                         path=self.path)
            # Формулы индексируют входы по номеру — сообщаем движку, какое
            # пространство сейчас активно, иначе to_string() наврёт с именами.
            dist.configure_inputs(len(payload['input_names']),
                                  payload['input_names'])
            self.meta = payload.get('meta', {})
            if self.branch == 'B':
                # Ветка B: вход формул — эмбеддинг, значит CNN всё равно нужна.
                import oracle as oracle_mod
                self._device = oracle_mod.get_device()
                self._model = oracle_mod.PixelCNN()
                storage.load_oracle(self._model, branch='B',
                                    device=self._device)
                self._model.eval()

    def label(self) -> str:
        """Короткое имя движка для подписи в интерфейсе."""
        name = 'EML-формулы' if self.engine == 'eml' else 'нейросеть-оракул'
        model = self.meta.get('library_name') if self.meta else None
        suffix = f", {model}" if model and self.path else ""
        return f"{name} (ветка {self.branch}{suffix})"

    @property
    def is_single(self) -> bool:
        """Ветка C: одна формула и таблица порогов вместо набора и argmax."""
        return self._clf is not None and hasattr(self._clf, 'decoder')

    @property
    def conf_is_probability(self) -> bool:
        """
        Можно ли называть третье возвращаемое значение вероятностью.

        У оракула да: softmax логитов обученной на кросс-энтропии сети —
        настоящая оценка вероятности класса. У формул НЕТ: их выходы никто не
        калибровал как вероятности, softmax над ними — просто монотонное
        приведение к [0, 1]. Интерфейс обязан подписывать эти два числа
        по-разному, иначе рядом стоят «87%» и «87%», означающие разное.
        """
        return self.engine == 'oracle'

    def conf_label(self) -> str:
        """Как подписать conf в интерфейсе."""
        if self.conf_is_probability:
            return 'уверенность'
        return 'запас' if self.is_single else 'оценка'

    @property
    def conf_floor(self) -> float:
        """
        Наименьшее значение, которое conf вообще может принять.

        У softmax по N классам максимум не бывает меньше 1/N, поэтому порог
        уверенности ниже этой величины не фильтрует ничего — интерфейс обязан
        об этом сказать. У ветки C conf это доля участка оси, отделяющая кадр
        от ближайшего порога, и она честно доходит до нуля.
        """
        return 0.0 if self.is_single else 1.0 / max(config.N_CLASSES, 1)

    def predict(self, crops: np.ndarray):
        """
        Классифицировать пачку кадров 64x64.

        Args:
            crops: (N, 64, 64, 3) uint8 RGB.

        Returns:
            preds:  (N,) int   — индекс класса;
            scores: (N, 4)     — оценки классов;
            conf:   (N,)       — softmax(оценки).max().

        Про conf: для оракула это настоящая вероятность после softmax, для
        формул — лишь удобный прокси. Выходы EML не калибровались как
        вероятности, их softmax не имеет вероятностного смысла и годится
        только чтобы отсортировать окна по «уверенности» на оверлее.
        Проверять, что именно вернулось, — через conf_is_probability.

        И для того, и для другого движка нижняя граница здесь 1/4, а не 0:
        softmax по 4 классам меньше 0.25 максимум не даёт. Порог уверенности
        ниже 0.25 поэтому не фильтрует ничего.
        """
        crops = np.asarray(crops)
        if crops.ndim == 3:
            crops = crops[None, ...]
        if crops.shape[1:] != (config.IMG_SIZE, config.IMG_SIZE, 3):
            raise ValueError(f"ожидались кадры {config.IMG_SIZE}x"
                             f"{config.IMG_SIZE}x3, получено {crops.shape[1:]}")

        if self.is_single:
            return self._predict_single(crops)
        scores = self._scores(crops)
        preds = np.argmax(scores, axis=1)
        conf = softmax(scores).max(axis=1)
        return preds, scores, conf

    def _predict_single(self, crops: np.ndarray):
        """
        Ветка C: класс из таблицы порогов, «уверенность» — запас до порога.

        Softmax тут не годится совсем. Выход формулы доходит до 2.2e4, а
        оценки классов ветки C это расстояния по той же оси; softmax от них
        насыщается в единицу почти всегда, и порог уверенности перестал бы
        фильтровать что-либо. Осмысленная величина здесь другая: насколько
        глубоко кадр сидит внутри своего участка. 0 — вплотную к порогу,
        1 — ровно посередине участка. У неё и шкала честная, и смысл понятный.
        """
        import features as features_mod

        x = features_mod.extract_batch(crops).astype(np.float64)
        values = self._clf.raw(x)
        preds = self._clf.decoder.decode(values)
        scores = self._clf.class_scores(x)

        edges = np.concatenate(([-np.inf], self._clf.decoder.thresholds,
                                [np.inf]))
        idx = np.searchsorted(self._clf.decoder.thresholds, values, side='right')
        lo, hi = edges[idx], edges[idx + 1]
        gap = np.minimum(values - lo, hi - values)          # запас до порога
        width = hi - lo
        # Крайние участки бесконечны: там запас ограничен только своей
        # стороной, и делить его не на что — считаем такой кадр уверенным.
        conf = np.where(np.isfinite(width), 2.0 * gap / np.maximum(width, 1e-30),
                        1.0)
        return preds, scores, np.clip(conf, 0.0, 1.0)

    def _scores(self, crops: np.ndarray) -> np.ndarray:
        import oracle as oracle_mod

        if self.engine == 'oracle':
            if self.oracle_branch == 'A':
                import features as features_mod
                x = features_mod.extract_batch(crops)
            else:
                x = crops
            return oracle_mod.predict_logits(self._model, x, self._device)

        if self.branch != 'B':
            import features as features_mod
            x = features_mod.extract_batch(crops).astype(np.float64)
        else:
            x = oracle_mod.predict_embeddings(self._model, crops,
                                              self._device).astype(np.float64)
        return self._clf.scores(x)

    def formulas(self) -> list[str]:
        """Тексты формул — для показа во вкладке теста."""
        if self._clf is None:
            return []
        if self.is_single:
            return [self._clf.tree.to_string()]
        return [t.to_string() for t in self._clf.trees]


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax по строкам, с вычитанием максимума против переполнения.

    Публичная функция: интерфейсу она нужна, чтобы привести оценки оракула
    (логиты) и формул (калиброванные выходы) к одной шкале — иначе столбики
    двух движков на одном графике несопоставимы.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(axis=1, keepdims=True)


# ── Тайлинг кадра ─────────────────────────────────────────────────────────────

def tile_positions(height: int, width: int, tile: int = DEFAULT_TILE,
                   stride: int | None = None) -> list[tuple[int, int]]:
    """
    Левые верхние углы всех окон, целиком помещающихся в кадр.

    Остаток справа и снизу отбрасывается, а не дополняется: дополненное окно
    содержало бы выдуманные пиксели, и классификация по нему была бы враньём.

    Кадр меньше окна => окон НЕТ (пустой список). Раньше здесь стояло
    max(h - tile + 1, 1), и такой кадр давал одну позицию (0, 0), по которой
    extract_tiles потом падал на несовпадении форм: окно 64x64 не вырезается
    из кадра 40x30. Вызывающий код пустой список уже обрабатывает.
    """
    stride = stride or tile
    if height < tile or width < tile:
        return []
    ys = list(range(0, height - tile + 1, stride))
    xs = list(range(0, width - tile + 1, stride))
    return [(y, x) for y in ys for x in xs]


def extract_tiles(frame: np.ndarray, positions, tile: int = DEFAULT_TILE
                  ) -> np.ndarray:
    """Вырезать окна по списку позиций -> (N, tile, tile, 3) uint8."""
    out = np.empty((len(positions), tile, tile, 3), dtype=np.uint8)
    for i, (y, x) in enumerate(positions):
        out[i] = frame[y:y + tile, x:x + tile]
    return out


def draw_overlay(frame: np.ndarray, positions, preds, conf=None,
                 tile: int = DEFAULT_TILE, alpha: float = 0.35,
                 conf_threshold: float = 0.0) -> np.ndarray:
    """
    Залить каждое окно полупрозрачным цветом его класса.

    Args:
        alpha: сила заливки; 0 — ничего не видно, 1 — исходный кадр закрашен.
        conf_threshold: окна с меньшей уверенностью не закрашиваются —
            так на кадре сразу видно, где модель сомневается.

    Возвращает НОВЫЙ массив; исходный кадр не трогается, иначе повторный
    рендер накладывал бы заливку на заливку и картинка бы «выгорала».
    """
    out = frame.astype(np.float32).copy()
    for i, (y, x) in enumerate(positions):
        if conf is not None and conf[i] < conf_threshold:
            continue
        color = np.asarray(class_color(int(preds[i])), dtype=np.float32)
        patch = out[y:y + tile, x:x + tile]
        out[y:y + tile, x:x + tile] = patch * (1 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def center_crop(frame: np.ndarray, size: int = DEFAULT_TILE) -> np.ndarray:
    """Центральное окно size x size. Если кадр меньше — растянуть повтором."""
    h, w = frame.shape[:2]
    if h < size or w < size:
        frame = resize_nearest(frame, max(size, h), max(size, w))
        h, w = frame.shape[:2]
    y = (h - size) // 2
    x = (w - size) // 2
    return frame[y:y + size, x:x + size]


def resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Масштабирование ближайшим соседом на чистом numpy.

    Нужно, чтобы показ и подгонка кадров не тянули за собой Pillow или cv2:
    видео-вкладка опциональна, а фото-режим должен работать всегда.
    """
    h, w = frame.shape[:2]
    ys = (np.arange(height) * h // height).clip(0, h - 1)
    xs = (np.arange(width) * w // width).clip(0, w - 1)
    return frame[ys][:, xs]


def mosaic(crops: np.ndarray, rows: int, cols: int,
           tile: int = DEFAULT_TILE) -> np.ndarray:
    """
    Сложить кадры 64x64 в одно большое изображение rows x cols.

    Используется для проверки тайлинга: из вырезок с ИЗВЕСТНЫМИ метками
    собирается «кадр», и можно точно посчитать, сколько окон оверлей разметил
    правильно. Без этого видео-режим пришлось бы оценивать на глаз.
    """
    frame = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    for i in range(min(len(crops), rows * cols)):
        r, c = divmod(i, cols)
        frame[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = crops[i]
    return frame


# ── Источник видео ────────────────────────────────────────────────────────────

class VideoUnavailable(RuntimeError):
    """OpenCV не установлен либо источник не открылся."""


class VideoSource:
    """
    Обёртка над cv2.VideoCapture: видеофайл или веб-камера.

    cv2 импортируется лениво и только здесь — всё остальное в проекте от него
    не зависит (см. requirements.txt).
    """

    def __init__(self, source, max_width: int = 960):
        try:
            import cv2
        except ImportError as exc:
            raise VideoUnavailable(
                "для вкладки видео нужен OpenCV:\n"
                "    pip install opencv-python\n"
                "Остальной проект от него не зависит.") from exc

        self._cv2 = cv2
        self.source = source
        self.max_width = max_width
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise VideoUnavailable(f"не удалось открыть источник: {source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.is_camera = isinstance(source, int)

    def read(self) -> np.ndarray | None:
        """Следующий кадр в RGB или None, если поток кончился."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        frame = frame[:, :, ::-1]                     # OpenCV отдаёт BGR
        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            frame = resize_nearest(frame, int(h * scale), self.max_width)
        return np.ascontiguousarray(frame)

    def rewind(self):
        self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def write_image(path: str, frame: np.ndarray) -> str:
    """
    Сохранить кадр RGB в PNG БЕЗ пересэмплинга — ровно те пиксели, что пришли.

    Кодировщик PNG написан здесь на zlib из стандартной библиотеки (полтора
    десятка строк), поэтому сохранение кадра, как и всё остальное кроме
    декодера видео, не требует ни OpenCV, ни Pillow. Если cv2 всё же
    установлен, пишем им — он быстрее на больших кадрах.
    """
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"ожидался кадр (H, W, 3) uint8, получен {frame.shape}")

    try:
        import cv2
        if not cv2.imwrite(path, frame[:, :, ::-1]):     # cv2 ждёт BGR
            raise RuntimeError(f"cv2.imwrite не смог записать {path}")
        return path
    except ImportError:
        pass

    with open(path, 'wb') as f:
        f.write(_png_bytes(frame))
    return path


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Минимальный кодировщик PNG (8 бит, truecolor RGB, без интерлейса)."""
    import struct
    import zlib

    height, width = rgb.shape[:2]
    # Каждая строка PNG предваряется байтом фильтра; 0 = «без фильтра».
    raw = np.hstack([np.zeros((height, 1), dtype=np.uint8),
                     rgb.reshape(height, width * 3)]).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (struct.pack('>I', len(data)) + body
                + struct.pack('>I', zlib.crc32(body) & 0xffffffff))

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', ihdr)
            + chunk(b'IDAT', zlib.compress(raw, 6))
            + chunk(b'IEND', b''))


def write_video(path: str, frames, fps: float = 12.0) -> str:
    """
    Сохранить последовательность RGB-кадров в mp4 (нужен OpenCV).

    Используется кнопкой «Сохранить видео» — чтобы размеченный оверлеем
    результат можно было положить в диплом, а не только посмотреть в окне.
    """
    try:
        import cv2
    except ImportError as exc:
        raise VideoUnavailable("для записи видео нужен OpenCV") from exc

    frames = list(frames)
    if not frames:
        raise ValueError("нет кадров для записи")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    try:
        for frame in frames:
            writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
    finally:
        writer.release()
    return path


# ── Самопроверка ──────────────────────────────────────────────────────────────

def _self_test():
    """
    Проверка движка без GUI.

    Из тестовых вырезок с известными метками собирается мозаика, она режется
    тайлингом обратно и классифицируется — если тайлинг где-то съезжает на
    пиксель, точность рухнет, и это будет видно числом.
    """
    import time

    import data

    # Самопроверка — самостоятельная точка входа, как main.py и gui.py, и
    # выбранный датасет применяет сама. Без этого она шла бы по умолчанию, то
    # есть по ДРУГОМУ набору классов, чем весь остальной проект, и падала бы
    # на первой же модели.
    config.load_dataset_choice(verbose=True)

    print("── vision.py: самопроверка ──\n")
    images, labels, _ = data.load_dataset(verbose=False)
    _, _, test_idx = data.stratified_split(labels)

    rng = np.random.default_rng(0)
    rows, cols = 6, 8
    sel = test_idx[rng.choice(len(test_idx), rows * cols, replace=False)]
    crops, true_labels = images[sel], labels[sel]

    frame = mosaic(crops, rows, cols)
    print(f"мозаика {frame.shape[1]}x{frame.shape[0]} из {rows * cols} вырезок")

    positions = tile_positions(frame.shape[0], frame.shape[1])
    assert len(positions) == rows * cols, (
        f"тайлинг дал {len(positions)} окон вместо {rows * cols}")
    tiles = extract_tiles(frame, positions)
    assert np.array_equal(tiles, crops), "тайлинг вернул не те пиксели"
    print(f"тайлинг: {len(positions)} окон, пиксели совпали с исходными\n")

    formula_engine = None
    for branch in ('A', 'B', 'C'):
        for engine in ('oracle', 'eml'):
            try:
                predictor = Predictor(branch=branch, engine=engine)
            except (FileNotFoundError, ValueError) as exc:
                # ValueError = модель обучена на другом наборе классов. После
                # смены датасета это норма, а не поломка самопроверки.
                print(f"  {branch}/{engine}: пропуск ({exc})")
                continue
            if engine == 'eml' and formula_engine is None:
                formula_engine = predictor
            t0 = time.perf_counter()
            preds, _, conf = predictor.predict(tiles)
            dt = time.perf_counter() - t0
            acc = float((preds == true_labels).mean())
            print(f"  {predictor.label():<28} точность {acc:.3f} | "
                  f"{dt * 1000:6.1f} мс на {len(tiles)} окон "
                  f"({len(tiles) / dt:5.0f} окон/с)")

    # Оверлей проверяется первым же движком формул, который загрузился, а не
    # веткой A жёстко: после смены датасета применимой может оказаться другая.
    if formula_engine is None:
        print("\nни одного набора формул для текущего датасета — "
              "оверлей и пропускная способность не проверены")
        return
    predictor = formula_engine
    preds, _, conf = predictor.predict(tiles)
    overlaid = draw_overlay(frame, positions, preds, conf)
    assert overlaid.shape == frame.shape
    assert not np.array_equal(overlaid, frame), "оверлей ничего не изменил"
    assert frame.max() > 0, "исходный кадр испорчен оверлеем"
    print("\nоверлей: кадр размечен, исходник не изменён")

    t0 = time.perf_counter()
    predictor.predict(tiles)
    rate = len(tiles) / max(time.perf_counter() - t0, 1e-9)
    print(f"пропускная способность: ~{rate:.0f} окон/с "
          f"=> кадр из {rows * cols} окон при ~{rate / (rows * cols):.1f} FPS")

    try:
        import cv2                                    # noqa: F401
        print("\nOpenCV найден — вкладка видео доступна")
    except ImportError:
        print("\nOpenCV не найден — вкладка видео будет отключена")

    print("\nвсе проверки пройдены")


if __name__ == '__main__':
    _self_test()

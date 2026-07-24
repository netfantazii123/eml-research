"""
data.py — загрузка датасета аэрофотоснимков и стратифицированный сплит.

BMP читается СВОИМ парсером на numpy, без Pillow/OpenCV. Это не упрямство:
весь датасет — 24-битные несжатые BMP с BITMAPINFOHEADER (проверено по
заголовкам всех 5049 файлов), для такого формата парсер занимает 30 строк.
Проект целится в микроконтроллер, поэтому лишние тяжёлые зависимости в
пайплайне нежелательны на любом этапе.

Первый вызов load_dataset() читает BMP-файлы и кладёт распакованный массив в
dataset/cache.npz (~62 МБ uint8). Дальше загрузка мгновенная.
"""

import os
import struct

import numpy as np

import config


# ── Парсер BMP ────────────────────────────────────────────────────────────────

def read_bmp(path: str) -> np.ndarray:
    """
    Прочитать 24-битный несжатый BMP в массив (H, W, 3) uint8 в порядке RGB.

    Поддерживается ровно тот подформат, в котором лежит датасет:
    сигнатура 'BM', bpp=24, compression=0 (BI_RGB). Всё остальное — ошибка,
    молча портить данные хуже, чем упасть.
    """
    with open(path, 'rb') as f:
        raw = f.read()

    if raw[:2] != b'BM':
        raise ValueError(f"{path}: не BMP (сигнатура {raw[:2]!r})")

    pixel_offset, = struct.unpack('<I', raw[10:14])
    width, height = struct.unpack('<ii', raw[18:26])
    bpp, = struct.unpack('<H', raw[28:30])
    compression, = struct.unpack('<I', raw[30:34])

    if bpp != 24 or compression != 0:
        raise ValueError(f"{path}: ожидался 24bpp BI_RGB, получено "
                         f"bpp={bpp} compression={compression}")

    # Отрицательная высота = строки сверху вниз; положительная = снизу вверх.
    top_down = height < 0
    height = abs(height)

    # Каждая строка выровнена до кратности 4 байтам.
    row_bytes = width * 3
    padded_row = (row_bytes + 3) & ~3

    buf = np.frombuffer(raw, dtype=np.uint8, count=padded_row * height,
                        offset=pixel_offset)
    img = buf.reshape(height, padded_row)[:, :row_bytes].reshape(height, width, 3)

    if not top_down:
        img = img[::-1]              # BMP хранит строки снизу вверх
    return img[:, :, ::-1].copy()    # BGR -> RGB


# ── Загрузка датасета ─────────────────────────────────────────────────────────

def dataset_fingerprint() -> str:
    """
    Дешёвый отпечаток содержимого dataset/: число файлов и суммарный размер.

    Полное хеширование 83 МБ на каждый запуск бессмысленно дорого, а вот
    «в папку добавили/убрали картинки» этой пары чисел достаточно. Без неё
    cache.npz не инвалидируется НИКОГДА: обновив датасет, пользователь молча
    продолжит учиться на старом.

    Пустая строка = посчитать не удалось (папок с классами нет). Это не ошибка:
    так выглядит машина, куда принесли только кэш, — и тогда кэш надо
    принять как есть, а не пытаться пересобрать его из несуществующих BMP.
    """
    n_files, total_bytes = 0, 0
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            return ''
        with os.scandir(class_dir) as it:
            for entry in it:
                if entry.name.lower().endswith('.bmp') and entry.is_file():
                    n_files += 1
                    total_bytes += entry.stat().st_size
    if n_files == 0:
        return ''
    return f"{n_files}:{total_bytes}"


def load_dataset(use_cache: bool = True, verbose: bool = True):
    """
    Загрузить весь датасет.

    Returns:
        images: (N, 64, 64, 3) uint8
        labels: (N,) int64 — индекс в config.CLASS_NAMES
        paths:  (N,) список относительных путей (для отчётов об ошибках)
    """
    cache = config.CACHE_PATH
    if use_cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        current = dataset_fingerprint()
        cached = str(d['fingerprint']) if 'fingerprint' in d.files else None
        # Пустой current = исходных BMP на этой машине нет, пересчитать нечем;
        # старый кэш без отпечатка принимаем, но предупреждаем.
        if not current or cached == current:
            if verbose:
                print(f"[data] кэш: {cache}")
                if cached is None:
                    print("[data] кэш без отпечатка (создан старой версией) — "
                          "будет обновлён при следующем пересчёте")
            return d['images'], d['labels'], list(d['paths'])
        if verbose:
            print(f"[data] датасет изменился ({cached} -> {current}) — "
                  f"перечитываю BMP")

    images, labels, paths = [], [], []
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"нет папки класса {class_dir}. Ожидается структура "
                f"{config.DATASET_DIR}/<CLASS>/*.bmp")
        files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith('.bmp'))
        if verbose:
            print(f"[data] {class_name:>7}: {len(files)} файлов")
        for name in files:
            img = read_bmp(os.path.join(class_dir, name))
            if img.shape != (config.IMG_SIZE, config.IMG_SIZE, 3):
                raise ValueError(f"{name}: размер {img.shape}, "
                                 f"ожидался {config.IMG_SIZE}x{config.IMG_SIZE}x3")
            images.append(img)
            labels.append(class_idx)
            paths.append(f"{class_name}/{name}")

    images = np.stack(images)
    labels = np.asarray(labels, dtype=np.int64)

    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    np.savez_compressed(cache, images=images, labels=labels,
                        paths=np.array(paths, dtype=object),
                        fingerprint=dataset_fingerprint())
    if verbose:
        print(f"[data] загружено {len(images)} картинок -> кэш {cache}")
    return images, labels, paths


# ── Сплиты ────────────────────────────────────────────────────────────────────

def stratified_split(labels: np.ndarray, seed: int | None = None):
    """
    Стратифицированный сплит train/val/test: доли классов сохраняются в каждой
    части. Возвращает три массива индексов.

    Стратификация здесь обязательна: BEACH втрое реже ROADS, при случайном
    сплите редкий класс может почти исчезнуть из валидации и метрика поедет.
    """
    if seed is None:
        seed = config.SPLIT_SEED
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for class_idx in range(config.N_CLASSES):
        idx = np.flatnonzero(labels == class_idx)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * config.SPLIT_TRAIN))
        n_val = int(round(n * config.SPLIT_VAL))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    out = tuple(np.sort(np.concatenate(parts))
                for parts in (train_idx, val_idx, test_idx))
    return out


def class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Веса классов 1/freq, отнормированные к среднему 1.

    Нужны из-за дисбаланса 3.4:1 — без них лосс доминируют ROADS и PEOPLE.
    """
    counts = np.bincount(labels, minlength=config.N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (config.N_CLASSES * counts)
    return (w / w.mean()).astype(np.float32)


def describe(labels: np.ndarray, name: str = 'dataset') -> str:
    """Строка вида 'train: 3534 (BEACH 428, MOTO 572, ...)' для логов."""
    counts = np.bincount(labels, minlength=config.N_CLASSES)
    parts = ', '.join(f"{c} {counts[i]}" for i, c in enumerate(config.CLASS_NAMES))
    return f"{name}: {len(labels)} ({parts})"


if __name__ == '__main__':
    images, labels, paths = load_dataset()
    tr, va, te = stratified_split(labels)
    print(describe(labels, 'всего'))
    print(describe(labels[tr], 'train'))
    print(describe(labels[va], 'val  '))
    print(describe(labels[te], 'test '))
    print('веса классов:', dict(zip(config.CLASS_NAMES,
                                    class_weights(labels[tr]).round(3))))

"""
data.py — загрузка датасета картинок по папкам-классам и стратифицированный сплит.

СТРУКТУРА ДАТАСЕТА — одна подпапка на класс:

    <любая папка>/
        BEACH/*.bmp      MOTO/*.png      PEOPLE/*.jpg      ROADS/...

Имена классов, их количество и порядок берутся из файловой системы, а не из
константы: папку выбирает пользователь. Порядок — алфавитный, потому что
индекс класса попадает и в веса оракула, и в таблицу порогов ветки C, и
зависеть от того, в каком порядке ОС отдала записи каталога, он не может.

BMP читается СВОИМ парсером на numpy (imaging.read_bmp), без Pillow/OpenCV.
Это не упрямство: исходный датасет — 24-битные несжатые BMP с
BITMAPINFOHEADER (проверено по заголовкам всех 5049 файлов), для такого
формата парсер занимает 30 строк, а проект целится в микроконтроллер, где
лишние тяжёлые зависимости нежелательны на любом этапе. Остальные форматы и
произвольные размеры поддержаны в imaging.py.

Первый вызов load_dataset() читает файлы и кладёт распакованный массив в
кэш (~62 МБ uint8 на исходном датасете). Дальше загрузка мгновенная.
"""

import os

import numpy as np

import config
import imaging
from imaging import IMAGE_EXTS, read_bmp, read_image     # noqa: F401 (ре-экспорт)


# ── Сканирование папки ────────────────────────────────────────────────────────

def _image_files(directory: str) -> list[str]:
    """Отсортированные имена файлов-картинок в папке (без рекурсии)."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(n for n in names
                  if n.lower().endswith(IMAGE_EXTS)
                  and os.path.isfile(os.path.join(directory, n)))


def scan_classes(root: str) -> list[str]:
    """
    Имена классов = подпапки root, в которых есть хотя бы одна картинка.

    Пустые подпапки пропускаются молча: класс без единого примера всё равно
    сломал бы и стратификацию, и сбалансированную accuracy (деление на ноль в
    recall), а чаще всего это просто забытая папка вроде `.ipynb_checkpoints`.
    """
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if name.startswith('.'):
            continue
        if _image_files(os.path.join(root, name)):
            out.append(name)
    return out


def describe_root(root: str) -> dict:
    """
    Что лежит в папке — до того, как её выберут.

    Возвращает имена классов, число файлов по классам, встреченные расширения
    и размеры первых картинок каждого класса. Нужен интерфейсу: показать это
    ДО запуска обучения дешевле, чем объяснять потом, почему модель училась
    на трёх классах вместо четырёх.
    """
    classes = scan_classes(root)
    counts, exts, sizes, total_bytes = {}, set(), set(), 0
    for class_name in classes:
        class_dir = os.path.join(root, class_name)
        files = _image_files(class_dir)
        counts[class_name] = len(files)
        for name in files:
            exts.add(os.path.splitext(name)[1].lower())
            total_bytes += os.path.getsize(os.path.join(class_dir, name))
        for name in files[:1]:
            try:
                img = imaging.read_image(os.path.join(class_dir, name))
                sizes.add(f"{img.shape[1]}x{img.shape[0]}")
            except (OSError, ValueError, RuntimeError):
                sizes.add('?')
    return {'root': root, 'classes': classes, 'counts': counts,
            'n_files': sum(counts.values()), 'total_bytes': total_bytes,
            'exts': sorted(exts), 'sizes': sorted(sizes)}


# ── Загрузка датасета ─────────────────────────────────────────────────────────

def dataset_fingerprint() -> str:
    """
    Дешёвый отпечаток содержимого датасета: классы, число файлов, суммарный размер.

    Полное хеширование 83 МБ на каждый запуск бессмысленно дорого, а вот «в
    папку добавили/убрали картинки» или «выбрали другой набор классов» этих
    трёх величин достаточно. Без отпечатка кэш не инвалидируется НИКОГДА:
    сменив датасет, пользователь молча продолжит учиться на старом.

    Имена классов входят в отпечаток обязательно: при переключении на папку с
    тем же числом файлов (например, переименованную копию) размер и количество
    совпали бы, а метки означали бы совсем другое.

    Пустая строка = посчитать не удалось (папок с классами нет). Это не ошибка:
    так выглядит машина, куда принесли только кэш, — и тогда кэш надо принять
    как есть, а не пытаться пересобрать его из несуществующих файлов.
    """
    n_files, total_bytes = 0, 0
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            return ''
        for name in _image_files(class_dir):
            n_files += 1
            total_bytes += os.path.getsize(os.path.join(class_dir, name))
    if n_files == 0:
        return ''
    return f"{'|'.join(config.CLASS_NAMES)}:{n_files}:{total_bytes}"


def load_dataset(use_cache: bool = True, verbose: bool = True):
    """
    Загрузить весь датасет.

    Returns:
        images: (N, IMG_SIZE, IMG_SIZE, 3) uint8
        labels: (N,) int64 — индекс в config.CLASS_NAMES
        paths:  (N,) список относительных путей (для отчётов об ошибках)

    Картинка не того размера кропится по центру и масштабируется, а не
    растягивается: растяжение изменило бы масштаб текстуры, на который
    опираются coherence, block_std и другие признаки.
    """
    cache = config.CACHE_PATH
    if use_cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        current = dataset_fingerprint()
        cached = str(d['fingerprint']) if 'fingerprint' in d.files else None
        # Пустой current = исходных картинок на этой машине нет, пересчитать
        # нечем; старый кэш без отпечатка принимаем, но предупреждаем.
        if not current or cached == current:
            if verbose:
                print(f"[data] кэш: {cache}")
                if cached is None:
                    print("[data] кэш без отпечатка (создан старой версией) — "
                          "будет обновлён при следующем пересчёте")
            return d['images'], d['labels'], list(d['paths'])
        if verbose:
            print(f"[data] датасет изменился ({cached} -> {current}) — "
                  f"перечитываю картинки")

    images, labels, paths, resized = [], [], [], 0
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"нет папки класса {class_dir}. Ожидается структура "
                f"{config.DATASET_DIR}/<ИМЯ КЛАССА>/*{'|*'.join(IMAGE_EXTS)}")
        files = _image_files(class_dir)
        if verbose:
            print(f"[data] {class_name:>12}: {len(files)} файлов")
        for name in files:
            img = imaging.read_image(os.path.join(class_dir, name))
            if img.shape != (config.IMG_SIZE, config.IMG_SIZE, 3):
                img = imaging.fit_to_size(img, config.IMG_SIZE)
                resized += 1
            images.append(img)
            labels.append(class_idx)
            paths.append(f"{class_name}/{name}")

    if not images:
        raise FileNotFoundError(
            f"в {config.DATASET_DIR} не найдено ни одной картинки классов "
            f"{', '.join(config.CLASS_NAMES)}")

    images = np.stack(images)
    labels = np.asarray(labels, dtype=np.int64)

    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    np.savez_compressed(cache, images=images, labels=labels,
                        paths=np.array(paths, dtype=object),
                        fingerprint=dataset_fingerprint())
    if verbose:
        if resized:
            print(f"[data] приведено к {config.IMG_SIZE}x{config.IMG_SIZE}: "
                  f"{resized} из {len(images)} (кроп по центру + масштаб)")
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

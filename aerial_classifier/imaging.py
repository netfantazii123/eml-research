"""
imaging.py — чтение картинок и приведение их к входу модели.

Датасет проекта — 24-битные несжатые BMP 64x64, и парсер для них написан здесь
вручную (30 строк на numpy). Но выбирать папку с классами теперь может сам
пользователь, а в чужой папке лежит что угодно: png, jpg, другой размер,
чёрно-белое. Поэтому модуль устроен в три слоя:

  1. BMP — всегда свой парсер. Это формат датасета, на нём стоят тесты, и
     зависеть от сторонней библиотеки в основном пути загрузки не хочется.
  2. PNG — свой декодер на zlib из стандартной библиотеки (симметричен
     кодировщику в vision.py). Медленнее OpenCV, но работает всегда.
  3. JPEG — только через OpenCV или Pillow, если они есть. Декодер JPEG с нуля
     это тысячи строк, и писать его ради удобства выбора папки бессмысленно;
     вместо этого — внятное сообщение, чего не хватает.

Если OpenCV или Pillow установлены, они используются и для PNG тоже: на
больших картинках разница в скорости на порядки, а результат тот же.

Приведение к входу модели (`fit_to_size`) — это ЦЕНТРАЛЬНЫЙ КРОП плюс
масштабирование, а не растяжение. Растяжение изменило бы соотношение сторон, а
вместе с ним и масштаб текстуры, на который опираются coherence, block_std и
другие признаки; уменьшение усреднением по площади (а не выбором каждого
n-го пикселя) нужно ровно затем же — прореживание превратило бы мелкую
текстуру в шум муара, и признаки текстуры стали бы измерять артефакт.
"""

import os
import struct

import numpy as np

# Расширения, которые загрузчик датасета считает картинками.
IMAGE_EXTS = ('.bmp', '.png', '.jpg', '.jpeg')

_BACKEND = None          # None = ещё не искали; False = нет ни одного


# ── Необязательные быстрые декодеры ───────────────────────────────────────────

def _backend():
    """
    Первый доступный сторонний декодер: ('cv2'|'pil', модуль) или None.

    Ищется один раз и запоминается: попытка импорта отсутствующего модуля
    стоит обращения к файловой системе, а вызывается это на каждую картинку.
    """
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = False
        try:
            import cv2
            _BACKEND = ('cv2', cv2)
        except ImportError:
            try:
                from PIL import Image
                _BACKEND = ('pil', Image)
            except ImportError:
                pass
    return _BACKEND or None


def backend_name() -> str:
    """Имя доступного стороннего декодера или 'нет' — для сообщений в интерфейсе."""
    backend = _backend()
    return backend[0] if backend else 'нет'


def _read_via_backend(path: str) -> np.ndarray | None:
    backend = _backend()
    if backend is None:
        return None
    kind, module = backend
    if kind == 'cv2':
        img = module.imread(path, module.IMREAD_COLOR)
        if img is None:
            return None
        return np.ascontiguousarray(img[:, :, ::-1])          # BGR -> RGB
    with module.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.uint8)


# ── BMP ───────────────────────────────────────────────────────────────────────

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


# ── PNG ───────────────────────────────────────────────────────────────────────

_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}     # тип цвета -> каналов


def read_png(path: str) -> np.ndarray:
    """
    Прочитать PNG (8/16 бит, без интерлейса) в (H, W, 3) uint8 RGB.

    Реализация на zlib из стандартной библиотеки — симметрично кодировщику в
    vision.py. Интерлейс Adam7 не поддержан сознательно: он требует семи
    отдельных проходов ради формата, который почти не встречается, и
    непонятая картинка должна падать, а не читаться наполовину.
    """
    import zlib

    with open(path, 'rb') as f:
        raw = f.read()
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f"{path}: не PNG (сигнатура {raw[:8]!r})")

    pos, idat, header, palette, trns = 8, [], None, None, None
    while pos + 8 <= len(raw):
        length, = struct.unpack('>I', raw[pos:pos + 4])
        ctype = raw[pos + 4:pos + 8]
        body = raw[pos + 8:pos + 8 + length]
        pos += 12 + length                       # 4 длина + 4 тип + данные + 4 CRC
        if ctype == b'IHDR':
            header = struct.unpack('>IIBBBBB', body)
        elif ctype == b'PLTE':
            palette = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)
        elif ctype == b'tRNS':
            trns = body
        elif ctype == b'IDAT':
            idat.append(body)
        elif ctype == b'IEND':
            break

    if header is None:
        raise ValueError(f"{path}: нет заголовка IHDR")
    width, height, bit_depth, color_type, _, _, interlace = header
    if interlace:
        raise ValueError(f"{path}: интерлейс Adam7 не поддерживается "
                         f"(поставьте opencv-python или Pillow)")
    if color_type not in _PNG_CHANNELS or bit_depth not in (8, 16):
        raise ValueError(f"{path}: PNG тип {color_type} глубина {bit_depth} "
                         f"не поддерживается (поставьте opencv-python или Pillow)")

    channels = _PNG_CHANNELS[color_type]
    sample_bytes = bit_depth // 8
    bpp = channels * sample_bytes                # байт на пиксель (шаг фильтра)
    stride = width * bpp

    data = zlib.decompress(b''.join(idat))
    expected = (stride + 1) * height
    if len(data) < expected:
        raise ValueError(f"{path}: данных {len(data)} байт, ожидалось {expected}")

    flat = _png_unfilter(np.frombuffer(data, dtype=np.uint8, count=expected),
                         height, stride, bpp)

    if bit_depth == 16:
        # Старший байт каждого сэмпла: 16 бит на канал модель всё равно не ест.
        pixels = flat.reshape(height, width, channels, 2)[:, :, :, 0]
    else:
        pixels = flat.reshape(height, width, channels)

    if color_type == 3:                          # палитра
        if palette is None:
            raise ValueError(f"{path}: тип 3 (палитра) без чанка PLTE")
        _ = trns                                 # прозрачность не нужна: RGB
        return palette[pixels[:, :, 0].astype(np.int32)].astype(np.uint8)
    return to_rgb(pixels)


def _png_unfilter(data: np.ndarray, height: int, stride: int,
                  bpp: int) -> np.ndarray:
    """
    Снять построчные фильтры PNG. Возвращает плоский массив height*stride.

    Фильтры None/Sub/Up векторизованы; Average и Paeth опираются и на левый, и
    на верхний пиксель одновременно, поэтому считаются побайтно. На картинках
    датасета (64x64) это доли миллисекунды; на многомегапиксельных фотографиях
    заметно, и ровно поэтому при наличии OpenCV/Pillow сюда не заходят вовсе.
    """
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        base = y * (stride + 1)
        ftype = int(data[base])
        row = data[base + 1:base + 1 + stride].astype(np.int32)

        if ftype == 0:
            cur = row
        elif ftype == 1:                          # Sub: слева
            cur = row.copy()
            # По каждой из bpp «дорожек» это накопительная сумма по модулю 256.
            for k in range(bpp):
                cur[k::bpp] = np.cumsum(cur[k::bpp]) & 0xFF
        elif ftype == 2:                          # Up: сверху
            cur = (row + prev) & 0xFF
        elif ftype in (3, 4):
            cur = row
            up = prev.astype(np.int32)
            for i in range(stride):
                a = cur[i - bpp] if i >= bpp else 0
                b = up[i]
                if ftype == 3:                    # Average
                    cur[i] = (cur[i] + ((a + b) >> 1)) & 0xFF
                else:                             # Paeth
                    c = up[i - bpp] if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    cur[i] = (cur[i] + pred) & 0xFF
        else:
            raise ValueError(f"неизвестный фильтр строки PNG: {ftype}")

        prev = cur.astype(np.uint8)
        out[y] = prev
    return out.reshape(-1)


# ── Общий вход ────────────────────────────────────────────────────────────────

def to_rgb(img: np.ndarray) -> np.ndarray:
    """(H, W), (H, W, 1|2|3|4) -> (H, W, 3) uint8. Альфа отбрасывается."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 2:                       # серый + альфа
        img = np.repeat(img[:, :, :1], 3, axis=2)
    elif img.shape[2] >= 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img.astype(np.uint8))


def read_image(path: str) -> np.ndarray:
    """
    Прочитать картинку любого поддерживаемого формата в (H, W, 3) uint8 RGB.

    BMP всегда идёт через свой парсер: это формат датасета, и подменять его
    сторонним декодером значит терять контроль над самым горячим путём.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.bmp':
        return to_rgb(read_bmp(path))

    img = _read_via_backend(path)
    if img is not None:
        return to_rgb(img)
    if ext == '.png':
        return read_png(path)
    raise RuntimeError(
        f"{os.path.basename(path)}: для формата {ext or '?'} нужен OpenCV или "
        f"Pillow (pip install opencv-python). Без них читаются только .bmp и "
        f".png")


def center_crop_square(img: np.ndarray) -> np.ndarray:
    """Наибольший центральный квадрат. Крадёт края, но не искажает пропорции."""
    height, width = img.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return img[top:top + side, left:left + side]


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """
    Масштабировать квадратную картинку к size x size.

    Уменьшение — усреднением по площади: прореживание («каждый n-й пиксель»)
    на аэроснимке превращает регулярную текстуру в муар, а признаки текстуры
    начинают мерить артефакт масштабирования вместо самой текстуры.
    Увеличение — билинейно.
    """
    src = img.shape[0]
    if src == size:
        return np.ascontiguousarray(img.astype(np.uint8))

    work = img.astype(np.float32)
    if src > size:
        edges = np.linspace(0, src, size + 1).astype(np.int64)
        edges[-1] = src
        # Границы могут схлопнуться при почти равных размерах — тогда бин
        # шириной ноль дал бы деление на ноль.
        edges[1:] = np.maximum(edges[1:], edges[:-1] + 1)
        edges[-1] = src
        counts = np.diff(edges).astype(np.float32)
        rows = np.add.reduceat(work, edges[:-1], axis=0) / counts[:, None, None]
        out = np.add.reduceat(rows, edges[:-1], axis=1) / counts[None, :, None]
    else:
        pos = np.linspace(0, src - 1, size)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, src - 1)
        frac = (pos - lo).astype(np.float32)[:, None, None]
        rows = work[lo] * (1 - frac) + work[hi] * frac
        frac = frac.reshape(1, size, 1)
        out = rows[:, lo] * (1 - frac) + rows[:, hi] * frac

    return np.ascontiguousarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def fit_to_size(img: np.ndarray, size: int) -> np.ndarray:
    """Картинка любого размера -> (size, size, 3) uint8: кроп по центру + масштаб."""
    return resize(center_crop_square(to_rgb(img)), size)


def load_as_input(path: str, size: int) -> np.ndarray:
    """Прочитать файл и привести к входу модели одним вызовом."""
    return fit_to_size(read_image(path), size)

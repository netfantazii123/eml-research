"""
imaging.py — reading images and converting them to the model input.

The project dataset consists of 24-bit uncompressed 64x64 BMP files, and the
parser for them is written here by hand (30 lines of numpy). But the user can
now select the class folder, and a foreign folder may hold anything: png, jpg, a
different size, grayscale. The module is therefore organized in three layers:

  1. BMP — always the own parser. This is the dataset format, the tests rely on
     it, and depending on a third-party library on the main loading path is
     undesirable.
  2. PNG — an own decoder over zlib from the standard library (symmetric to the
     encoder in vision.py). Slower than OpenCV, but always available.
  3. JPEG — only through OpenCV or Pillow, if present. A JPEG decoder from
     scratch is thousands of lines; instead there is a clear message about what
     is missing.

If OpenCV or Pillow is installed, it is used for PNG as well: on large images
the speed difference is orders of magnitude, and the result is the same.

Conversion to the model input (`fit_to_size`) is a CENTER CROP plus scaling, not
a stretch. Stretching would change the aspect ratio and with it the texture
scale that coherence, block_std and other features rely on; downscaling by area
averaging (rather than taking every n-th pixel) serves exactly the same purpose
— decimation would turn fine texture into moiré noise, and the texture features
would start measuring the artifact.
"""

import os
import struct

import numpy as np

# Extensions the dataset loader treats as images.
IMAGE_EXTS = ('.bmp', '.png', '.jpg', '.jpeg')

_BACKEND = None          # None = not looked up yet; False = none available


# ── Optional fast decoders ────────────────────────────────────────────────────

def _backend():
    """
    The first available third-party decoder: ('cv2'|'pil', module) or None.

    Looked up once and remembered: attempting to import a missing module costs a
    file system access, and this is called for every image.
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
    """Name of the available third-party decoder or 'none' — for GUI messages."""
    backend = _backend()
    return backend[0] if backend else 'none'


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
    Read a 24-bit uncompressed BMP into an (H, W, 3) uint8 array in RGB order.

    Exactly the sub-format the dataset uses is supported: signature 'BM',
    bpp=24, compression=0 (BI_RGB). Anything else is an error — silently
    corrupting the data is worse than failing.
    """
    with open(path, 'rb') as f:
        raw = f.read()

    if raw[:2] != b'BM':
        raise ValueError(f"{path}: not a BMP (signature {raw[:2]!r})")

    pixel_offset, = struct.unpack('<I', raw[10:14])
    width, height = struct.unpack('<ii', raw[18:26])
    bpp, = struct.unpack('<H', raw[28:30])
    compression, = struct.unpack('<I', raw[30:34])

    if bpp != 24 or compression != 0:
        raise ValueError(f"{path}: expected 24bpp BI_RGB, got "
                         f"bpp={bpp} compression={compression}")

    # Negative height = rows top-down; positive = bottom-up.
    top_down = height < 0
    height = abs(height)

    # Every row is padded to a multiple of 4 bytes.
    row_bytes = width * 3
    padded_row = (row_bytes + 3) & ~3

    buf = np.frombuffer(raw, dtype=np.uint8, count=padded_row * height,
                        offset=pixel_offset)
    img = buf.reshape(height, padded_row)[:, :row_bytes].reshape(height, width, 3)

    if not top_down:
        img = img[::-1]              # BMP stores rows bottom-up
    return img[:, :, ::-1].copy()    # BGR -> RGB


# ── PNG ───────────────────────────────────────────────────────────────────────

_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}     # color type -> channels


def read_png(path: str) -> np.ndarray:
    """
    Read a PNG (8/16 bit, non-interlaced) into (H, W, 3) uint8 RGB.

    Implemented over zlib from the standard library — symmetric to the encoder
    in vision.py. Adam7 interlacing is deliberately unsupported: it requires
    seven separate passes for a format that is almost never encountered, and an
    image that is not understood must fail rather than be read halfway.
    """
    import zlib

    with open(path, 'rb') as f:
        raw = f.read()
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f"{path}: not a PNG (signature {raw[:8]!r})")

    pos, idat, header, palette, trns = 8, [], None, None, None
    while pos + 8 <= len(raw):
        length, = struct.unpack('>I', raw[pos:pos + 4])
        ctype = raw[pos + 4:pos + 8]
        body = raw[pos + 8:pos + 8 + length]
        pos += 12 + length                       # 4 length + 4 type + data + 4 CRC
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
        raise ValueError(f"{path}: no IHDR header")
    width, height, bit_depth, color_type, _, _, interlace = header
    if interlace:
        raise ValueError(f"{path}: Adam7 interlacing is not supported "
                         f"(install opencv-python or Pillow)")
    if color_type not in _PNG_CHANNELS or bit_depth not in (8, 16):
        raise ValueError(f"{path}: PNG type {color_type} depth {bit_depth} "
                         f"is not supported (install opencv-python or Pillow)")

    channels = _PNG_CHANNELS[color_type]
    sample_bytes = bit_depth // 8
    bpp = channels * sample_bytes                # bytes per pixel (filter step)
    stride = width * bpp

    data = zlib.decompress(b''.join(idat))
    expected = (stride + 1) * height
    if len(data) < expected:
        raise ValueError(f"{path}: {len(data)} bytes of data, expected {expected}")

    flat = _png_unfilter(np.frombuffer(data, dtype=np.uint8, count=expected),
                         height, stride, bpp)

    if bit_depth == 16:
        # High byte of each sample: the model does not consume 16 bits per
        # channel anyway.
        pixels = flat.reshape(height, width, channels, 2)[:, :, :, 0]
    else:
        pixels = flat.reshape(height, width, channels)

    if color_type == 3:                          # palette
        if palette is None:
            raise ValueError(f"{path}: type 3 (palette) without a PLTE chunk")
        _ = trns                                 # transparency not needed: RGB
        return palette[pixels[:, :, 0].astype(np.int32)].astype(np.uint8)
    return to_rgb(pixels)


def _png_unfilter(data: np.ndarray, height: int, stride: int,
                  bpp: int) -> np.ndarray:
    """
    Undo the PNG per-row filters. Returns a flat array of height*stride.

    Filters None/Sub/Up are vectorized; Average and Paeth depend on the left and
    the upper pixel simultaneously and are therefore computed byte by byte. On
    dataset images (64x64) this takes fractions of a millisecond; on
    multi-megapixel photographs it is noticeable, which is exactly why this path
    is not taken at all when OpenCV/Pillow is available.
    """
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        base = y * (stride + 1)
        ftype = int(data[base])
        row = data[base + 1:base + 1 + stride].astype(np.int32)

        if ftype == 0:
            cur = row
        elif ftype == 1:                          # Sub: from the left
            cur = row.copy()
            # Along each of the bpp "lanes" this is a cumulative sum mod 256.
            for k in range(bpp):
                cur[k::bpp] = np.cumsum(cur[k::bpp]) & 0xFF
        elif ftype == 2:                          # Up: from above
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
            raise ValueError(f"unknown PNG row filter: {ftype}")

        prev = cur.astype(np.uint8)
        out[y] = prev
    return out.reshape(-1)


# ── Common entry point ────────────────────────────────────────────────────────

def to_rgb(img: np.ndarray) -> np.ndarray:
    """(H, W), (H, W, 1|2|3|4) -> (H, W, 3) uint8. Alpha is discarded."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 2:                       # gray + alpha
        img = np.repeat(img[:, :, :1], 3, axis=2)
    elif img.shape[2] >= 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img.astype(np.uint8))


def read_image(path: str) -> np.ndarray:
    """
    Read an image of any supported format into (H, W, 3) uint8 RGB.

    BMP always goes through the own parser: it is the dataset format, and
    replacing it with a third-party decoder would mean losing control of the
    hottest path.
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
        f"{os.path.basename(path)}: format {ext or '?'} requires OpenCV or "
        f"Pillow (pip install opencv-python). Without them only .bmp and .png "
        f"are readable")


def center_crop_square(img: np.ndarray) -> np.ndarray:
    """Largest central square. Loses the edges but does not distort proportions."""
    height, width = img.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return img[top:top + side, left:left + side]


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """
    Scale a square image to size x size.

    Downscaling uses area averaging: decimation ("every n-th pixel") turns the
    regular texture of an aerial frame into moiré, and the texture features start
    measuring the scaling artifact instead of the texture itself.
    Upscaling is bilinear.
    """
    src = img.shape[0]
    if src == size:
        return np.ascontiguousarray(img.astype(np.uint8))

    work = img.astype(np.float32)
    if src > size:
        edges = np.linspace(0, src, size + 1).astype(np.int64)
        edges[-1] = src
        # The boundaries can collapse at nearly equal sizes — a zero-width bin
        # would then cause a division by zero.
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
    """Image of any size -> (size, size, 3) uint8: center crop + scaling."""
    return resize(center_crop_square(to_rgb(img)), size)

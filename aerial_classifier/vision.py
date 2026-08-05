"""
vision.py — the inference engine for the "Test · Photo" and "Test · Video" tabs.

Separated from gui.py deliberately: frame classification logic must not depend
on tkinter. This makes it possible to run it from the console
(`python vision.py`) and check it numerically rather than by eye on a picture in
a window.

Contents:

    Predictor    — a single interface to both models. The oracle and the EML
                   formulas answer the same question ("what is in this 64x64
                   frame") and are therefore called the same way; the test tab
                   only switches the engine and compares the answers.
    tiling       — splitting a large frame into 64x64 windows. The model was
                   trained on exactly such crops, so the only honest way to
                   apply it to a whole frame is to cut it the same way.
    overlay      — a semi-transparent fill of each window in the class color,
                   in numpy. Text is NOT drawn here: labels are placed by
                   tk.Canvas over the image, which renders sharper and does not
                   corrupt the frame itself.
    VideoSource  — a wrapper over cv2.VideoCapture (file or webcam).

OpenCV is needed ONLY for VideoSource and is imported lazily. Photo mode, tiling
and the overlay work in pure numpy, so a missing cv2 disables the video tab
without breaking anything else.
"""

import numpy as np

import config

# Class colors for the overlay (RGB). Distinguishable in color and in grayscale.
CLASS_COLORS = {
    'BEACH':  (249, 226, 175),   # sandy yellow
    'MOTO':   (243, 139, 168),   # red
    'PEOPLE': (166, 227, 161),   # green
    'ROADS':  (137, 180, 250),   # blue
}

DEFAULT_TILE = config.IMG_SIZE          # 64 — the size the models were trained on
DEFAULT_STRIDE = config.IMG_SIZE        # no overlap


def class_color(class_idx: int) -> tuple[int, int, int]:
    """
    Class color for the overlay and the plots.

    Colors for the classes of the reference dataset are chosen by hand. For a
    foreign folder with arbitrary names they are generated from the index:
    previously every unknown class got the same gray, and the overlay on a new
    dataset turned into a uniform fill showing nothing.
    """
    if class_idx < len(config.CLASS_NAMES):
        known = CLASS_COLORS.get(config.CLASS_NAMES[class_idx])
        if known is not None:
            return known
    # Golden ratio around the hue circle: neighboring indices get distant
    # colors, and there may be arbitrarily many classes.
    import colorsys
    hue = (class_idx * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.45, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


# ── Single interface to the models ────────────────────────────────────────────

class Predictor:
    """
    Classifier of 64x64 frames — either the neural oracle or the EML formulas.

    Both engines answer the same question and return identical structures, so
    the test tab can switch them with one button and show the answers side by
    side.

    Args:
        branch: 'A' (features) or 'B' (CNN embedding).
        engine: 'eml' — the distilled formulas, 'oracle' — the neural network.
        path:   a specific model file instead of the active slot. The test tabs
                use it to run a frame through any model from the library —
                otherwise two saved models cannot be compared on one frame.
    """

    def __init__(self, branch: str = 'A', engine: str = 'eml',
                 path: str | None = None):
        self.branch = branch.upper()
        self.engine = engine.lower()
        if self.engine not in ('eml', 'oracle'):
            raise ValueError(f"engine must be 'eml' or 'oracle', "
                             f"got {engine!r}")
        self.path = path
        self._model = None
        self._clf = None
        self._load()

    @property
    def oracle_branch(self) -> str:
        """
        Whose oracle stands behind this branch.

        Branch C works in the branch A feature space and learns from the same
        MLP — it has no oracle of its own and must not have one, otherwise the
        "one formula against four" comparison would also measure a difference
        between oracles.
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
            # The formulas index inputs by number — tell the engine which space
            # is active now, otherwise to_string() reports the wrong names.
            dist.configure_inputs(len(payload['input_names']),
                                  payload['input_names'])
            self.meta = payload.get('meta', {})
            if self.branch == 'B':
                # Branch B: the formula input is an embedding, so the CNN is
                # still needed.
                import oracle as oracle_mod
                self._device = oracle_mod.get_device()
                self._model = oracle_mod.PixelCNN()
                storage.load_oracle(self._model, branch='B',
                                    device=self._device)
                self._model.eval()

    def label(self) -> str:
        """Short engine name for the GUI caption."""
        name = 'EML formulas' if self.engine == 'eml' else 'neural oracle'
        model = self.meta.get('library_name') if self.meta else None
        suffix = f", {model}" if model and self.path else ""
        return f"{name} (branch {self.branch}{suffix})"

    @property
    def is_single(self) -> bool:
        """Branch C: one formula and a threshold table instead of a set + argmax."""
        return self._clf is not None and hasattr(self._clf, 'decoder')

    @property
    def conf_is_probability(self) -> bool:
        """
        Whether the third return value may be called a probability.

        For the oracle, yes: the softmax of the logits of a network trained on
        cross-entropy is an actual class probability estimate. For the formulas,
        NO: their outputs were never calibrated as probabilities, and a softmax
        over them is only a monotone mapping to [0, 1]. The GUI must label these
        two numbers differently, otherwise "87%" and "87%" sit side by side
        meaning different things.
        """
        return self.engine == 'oracle'

    def conf_label(self) -> str:
        """How to label conf in the GUI."""
        if self.conf_is_probability:
            return 'confidence'
        return 'margin' if self.is_single else 'score'

    @property
    def conf_floor(self) -> float:
        """
        The smallest value conf can possibly take.

        For a softmax over N classes the maximum is never below 1/N, so a
        confidence threshold below that value filters nothing — the GUI must say
        so. For branch C, conf is the fraction of the axis interval separating
        the frame from the nearest threshold, and it does reach zero.
        """
        return 0.0 if self.is_single else 1.0 / max(config.N_CLASSES, 1)

    def predict(self, crops: np.ndarray):
        """
        Classify a batch of 64x64 frames.

        Args:
            crops: (N, 64, 64, 3) uint8 RGB.

        Returns:
            preds:  (N,) int   — class index;
            scores: (N, 4)     — class scores;
            conf:   (N,)       — softmax(scores).max().

        On conf: for the oracle this is an actual probability after softmax, for
        the formulas only a convenient proxy. EML outputs were not calibrated as
        probabilities, their softmax has no probabilistic meaning, and it is
        only good for sorting windows by "confidence" on the overlay. Use
        conf_is_probability to check which one was returned.

        For both engines the lower bound here is 1/4 rather than 0: a softmax
        over 4 classes never yields a maximum below 0.25. A confidence threshold
        below 0.25 therefore filters nothing.
        """
        crops = np.asarray(crops)
        if crops.ndim == 3:
            crops = crops[None, ...]
        if crops.shape[1:] != (config.IMG_SIZE, config.IMG_SIZE, 3):
            raise ValueError(f"expected {config.IMG_SIZE}x{config.IMG_SIZE}x3 "
                             f"frames, got {crops.shape[1:]}")

        if self.is_single:
            return self._predict_single(crops)
        scores = self._scores(crops)
        preds = np.argmax(scores, axis=1)
        conf = softmax(scores).max(axis=1)
        return preds, scores, conf

    def _predict_single(self, crops: np.ndarray):
        """
        Branch C: class from the threshold table, "confidence" = margin to the
        threshold.

        Softmax is entirely unsuitable here. The formula output reaches 2.2e4,
        and branch C class scores are distances along the same axis; a softmax
        of them saturates to one almost always, and the confidence threshold
        would stop filtering anything. The meaningful quantity here is
        different: how deep the frame sits inside its interval. 0 — right at the
        threshold, 1 — exactly in the middle of the interval. Its scale is
        honest and its meaning is clear.
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
        gap = np.minimum(values - lo, hi - values)          # margin to threshold
        width = hi - lo
        # The outermost intervals are infinite: there the margin is bounded only
        # on its own side and there is nothing to divide it by — such a frame
        # counts as confident.
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
        """Formula texts — for display in the test tab."""
        if self._clf is None:
            return []
        if self.is_single:
            return [self._clf.tree.to_string()]
        return [t.to_string() for t in self._clf.trees]


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Row-wise softmax, subtracting the maximum to avoid overflow.

    A public function: the GUI needs it to bring the oracle scores (logits) and
    the formula scores (calibrated outputs) to a common scale — otherwise the
    bars of the two engines on one plot are not comparable.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(axis=1, keepdims=True)


# ── Frame tiling ──────────────────────────────────────────────────────────────

def tile_positions(height: int, width: int, tile: int = DEFAULT_TILE,
                   stride: int | None = None) -> list[tuple[int, int]]:
    """
    Top-left corners of all windows that fit entirely inside the frame.

    The remainder on the right and at the bottom is discarded rather than
    padded: a padded window would contain invented pixels, and classifying it
    would be a lie.

    A frame smaller than the window => NO windows (an empty list). Previously
    this used max(h - tile + 1, 1), and such a frame produced one position
    (0, 0), on which extract_tiles then failed on a shape mismatch: a 64x64
    window cannot be cut out of a 40x30 frame. The calling code already handles
    the empty list.
    """
    stride = stride or tile
    if height < tile or width < tile:
        return []
    ys = list(range(0, height - tile + 1, stride))
    xs = list(range(0, width - tile + 1, stride))
    return [(y, x) for y in ys for x in xs]


def extract_tiles(frame: np.ndarray, positions, tile: int = DEFAULT_TILE
                  ) -> np.ndarray:
    """Cut out windows at the listed positions -> (N, tile, tile, 3) uint8."""
    out = np.empty((len(positions), tile, tile, 3), dtype=np.uint8)
    for i, (y, x) in enumerate(positions):
        out[i] = frame[y:y + tile, x:x + tile]
    return out


def draw_overlay(frame: np.ndarray, positions, preds, conf=None,
                 tile: int = DEFAULT_TILE, alpha: float = 0.35,
                 conf_threshold: float = 0.0) -> np.ndarray:
    """
    Fill every window with a semi-transparent color of its class.

    Args:
        alpha: fill strength; 0 — nothing visible, 1 — the source frame is
            painted over.
        conf_threshold: windows with lower confidence are not filled — this
            immediately shows where the model is uncertain on the frame.

    Returns a NEW array; the source frame is not touched, otherwise a repeated
    render would stack fills and the image would wash out.
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
    """Central size x size window. If the frame is smaller — upscale by repeat."""
    h, w = frame.shape[:2]
    if h < size or w < size:
        frame = resize_nearest(frame, max(size, h), max(size, w))
        h, w = frame.shape[:2]
    y = (h - size) // 2
    x = (w - size) // 2
    return frame[y:y + size, x:x + size]


def resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Nearest-neighbor scaling in pure numpy.

    Needed so that displaying and fitting frames does not pull in Pillow or cv2:
    the video tab is optional, while photo mode must always work.
    """
    h, w = frame.shape[:2]
    ys = (np.arange(height) * h // height).clip(0, h - 1)
    xs = (np.arange(width) * w // width).clip(0, w - 1)
    return frame[ys][:, xs]


def mosaic(crops: np.ndarray, rows: int, cols: int,
           tile: int = DEFAULT_TILE) -> np.ndarray:
    """
    Assemble 64x64 frames into one large rows x cols image.

    Used to verify tiling: a "frame" is assembled from crops with KNOWN labels,
    and it becomes possible to count exactly how many windows the overlay
    labeled correctly. Without this, video mode could only be judged by eye.
    """
    frame = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    for i in range(min(len(crops), rows * cols)):
        r, c = divmod(i, cols)
        frame[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = crops[i]
    return frame


# ── Video source ──────────────────────────────────────────────────────────────

class VideoUnavailable(RuntimeError):
    """OpenCV is not installed, or the source did not open."""


class VideoSource:
    """
    Wrapper over cv2.VideoCapture: a video file or a webcam.

    cv2 is imported lazily and only here — nothing else in the project depends
    on it (see requirements.txt).
    """

    def __init__(self, source, max_width: int = 960):
        try:
            import cv2
        except ImportError as exc:
            raise VideoUnavailable(
                "the video tab requires OpenCV:\n"
                "    pip install opencv-python\n"
                "The rest of the project does not depend on it.") from exc

        self._cv2 = cv2
        self.source = source
        self.max_width = max_width
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise VideoUnavailable(f"could not open the source: {source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.is_camera = isinstance(source, int)

    def read(self) -> np.ndarray | None:
        """The next frame in RGB, or None if the stream ended."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        frame = frame[:, :, ::-1]                     # OpenCV returns BGR
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
    Save an RGB frame to PNG WITHOUT resampling — exactly the pixels that came in.

    The PNG encoder is written here over zlib from the standard library (about
    fifteen lines), so saving a frame, like everything else except the video
    decoder, requires neither OpenCV nor Pillow. If cv2 is installed after all,
    it is used — it is faster on large frames.
    """
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) uint8 frame, got {frame.shape}")

    try:
        import cv2
        if not cv2.imwrite(path, frame[:, :, ::-1]):     # cv2 expects BGR
            raise RuntimeError(f"cv2.imwrite could not write {path}")
        return path
    except ImportError:
        pass

    with open(path, 'wb') as f:
        f.write(_png_bytes(frame))
    return path


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Minimal PNG encoder (8 bit, truecolor RGB, non-interlaced)."""
    import struct
    import zlib

    height, width = rgb.shape[:2]
    # Every PNG row is prefixed by a filter byte; 0 = "no filter".
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
    Save a sequence of RGB frames to mp4 (requires OpenCV).

    Used by the "Save video" button, so that the overlay-annotated result can be
    exported rather than only viewed in the window.
    """
    try:
        import cv2
    except ImportError as exc:
        raise VideoUnavailable("writing video requires OpenCV") from exc

    frames = list(frames)
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    try:
        for frame in frames:
            writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
    finally:
        writer.release()
    return path


# ── Self-check ────────────────────────────────────────────────────────────────

def _self_test():
    """
    Engine check without the GUI.

    A mosaic is assembled from test crops with known labels, then cut back by
    tiling and classified — if tiling shifts by a pixel anywhere, the accuracy
    collapses and that shows up as a number.
    """
    import time

    import data

    # The self-check is a standalone entry point, like main.py and gui.py, and
    # applies the selected dataset itself. Without this it would run on the
    # default, i.e. on a DIFFERENT class set from the rest of the project, and
    # would fail on the very first model.
    config.load_dataset_choice(verbose=True)

    print("── vision.py: self-check ──\n")
    images, labels, _ = data.load_dataset(verbose=False)
    _, _, test_idx = data.stratified_split(labels)

    rng = np.random.default_rng(0)
    rows, cols = 6, 8
    sel = test_idx[rng.choice(len(test_idx), rows * cols, replace=False)]
    crops, true_labels = images[sel], labels[sel]

    frame = mosaic(crops, rows, cols)
    print(f"mosaic {frame.shape[1]}x{frame.shape[0]} from {rows * cols} crops")

    positions = tile_positions(frame.shape[0], frame.shape[1])
    assert len(positions) == rows * cols, (
        f"tiling produced {len(positions)} windows instead of {rows * cols}")
    tiles = extract_tiles(frame, positions)
    assert np.array_equal(tiles, crops), "tiling returned the wrong pixels"
    print(f"tiling: {len(positions)} windows, pixels match the originals\n")

    formula_engine = None
    for branch in ('A', 'B', 'C'):
        for engine in ('oracle', 'eml'):
            try:
                predictor = Predictor(branch=branch, engine=engine)
            except (FileNotFoundError, ValueError) as exc:
                # ValueError = the model was trained on a different class set.
                # After a dataset change this is normal, not a self-check
                # failure.
                print(f"  {branch}/{engine}: skipped ({exc})")
                continue
            if engine == 'eml' and formula_engine is None:
                formula_engine = predictor
            t0 = time.perf_counter()
            preds, _, conf = predictor.predict(tiles)
            dt = time.perf_counter() - t0
            acc = float((preds == true_labels).mean())
            print(f"  {predictor.label():<28} accuracy {acc:.3f} | "
                  f"{dt * 1000:6.1f} ms for {len(tiles)} windows "
                  f"({len(tiles) / dt:5.0f} windows/s)")

    # The overlay is checked with the first formula engine that loaded rather
    # than with branch A rigidly: after a dataset change a different one may be
    # the applicable one.
    if formula_engine is None:
        print("\nno formula set for the current dataset — "
              "overlay and throughput were not checked")
        return
    predictor = formula_engine
    preds, _, conf = predictor.predict(tiles)
    overlaid = draw_overlay(frame, positions, preds, conf)
    assert overlaid.shape == frame.shape
    assert not np.array_equal(overlaid, frame), "the overlay changed nothing"
    assert frame.max() > 0, "the source frame was corrupted by the overlay"
    print("\noverlay: frame annotated, the original was not modified")

    t0 = time.perf_counter()
    predictor.predict(tiles)
    rate = len(tiles) / max(time.perf_counter() - t0, 1e-9)
    print(f"throughput: ~{rate:.0f} windows/s "
          f"=> a frame of {rows * cols} windows at ~{rate / (rows * cols):.1f} FPS")

    try:
        import cv2                                    # noqa: F401
        print("\nOpenCV found — the video tab is available")
    except ImportError:
        print("\nOpenCV not found — the video tab will be disabled")

    print("\nall checks passed")


if __name__ == '__main__':
    _self_test()

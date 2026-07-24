"""
Тесты генератора C-кода.

Сгенерированный заголовок — конечный артефакт проекта, и рассинхрон между
features.py и шаблоном в export.py самая вероятная поломка при правках:
Python считает одно, микроконтроллер другое, а заметно это только по
`main.py verify-c` — если его вообще запустят. Здесь проверяется то, что
можно проверить без gcc и без обученных моделей.
"""

import re

import numpy as np
import pytest

import config
import eml_distiller as dist
import export
import features


@pytest.fixture(autouse=True)
def inputs_configured():
    dist.configure_inputs(config.N_FEATURES, list(config.FEATURE_NAMES))


# ── Экстрактор признаков ──────────────────────────────────────────────────────

def test_features_header_has_no_unfilled_placeholders():
    """
    Ни одной незакрытой фигурной скобки формата.

    Забытый ключ в generate_features_header упал бы сам, а вот лишний
    плейсхолдер в шаблоне уехал бы в C как есть и сломал компиляцию уже у
    того, кто взял заголовок на устройство.
    """
    header = export.generate_features_header()
    assert not re.search(r'(?<!\{)\{[a-z_][a-z0-9_]*\}', header)


def test_features_header_uses_dedicated_center_edge_scale():
    """
    f[26] нормируется своей константой, а не общей _SCALE_GRAD_MAG.

    Тест держит на месте исправление масштаба center_edge_diff: разъехавшись,
    Python и C дали бы для этого признака разные числа — молча, потому что
    формы и типы совпадают.
    """
    header = export.generate_features_header()
    line = next(ln for ln in header.splitlines() if 'f[26]' in ln)
    assert f"{features._SCALE_CENTER_EDGE}f" in line
    assert f"{features._SCALE_GRAD_MAG}f" not in line


def test_features_header_carries_config_constants():
    """Пороги вписываются из config, а не дублируются в шаблоне руками."""
    header = export.generate_features_header()
    for value in (config.EDGE_THRESHOLD, config.DARK_THRESHOLD,
                  config.BRIGHT_THRESHOLD):
        assert str(value) in header
    assert f"#define AERIAL_N_FEATURES   {config.N_FEATURES}" in header
    assert f"#define AERIAL_IMG_SIZE     {config.IMG_SIZE}" in header
    assert f"#define AERIAL_ENTROPY_BINS {config.ENTROPY_BINS}" in header


def test_features_header_assigns_every_feature():
    """Каждый из 32 признаков должен получить значение в C-коде."""
    header = export.generate_features_header()
    for i in range(config.N_FEATURES):
        assert re.search(rf'f\[{i}\]\s*=', header), f"признак {i} не заполнен"


def test_radial_constants_match_python():
    """
    Константы радиального профиля в C совпадают с посчитанными в features.py.

    Они не берутся из config, а вычисляются по геометрии кадра — значит могут
    разъехаться при смене IMG_SIZE.
    """
    header = export.generate_features_header()
    assert f"{features._DIST_STD:.10f}" in header
    assert f"{export._dist_mean():.10f}" in header
    expected_mean = float(np.sqrt(
        (np.arange(config.IMG_SIZE) - (config.IMG_SIZE - 1) / 2.0) ** 2
        + ((np.arange(config.IMG_SIZE) - (config.IMG_SIZE - 1) / 2.0)[:, None]) ** 2
    ).mean())
    assert export._dist_mean() == pytest.approx(expected_mean, rel=1e-12)


# ── Формулы ───────────────────────────────────────────────────────────────────

def toy_classifier():
    trees = [
        dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=20),
                     right=dist.EMLNode('const', value=-0.367)),
        dist.EMLNode('var', var_idx=0),
        dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=1),
                     right=dist.EMLNode('var', var_idx=2)),
        dist.EMLNode('const', value=1.5),
    ]
    return dist.EMLClassifier(trees, offsets=[0.1, 0.2, 0.3, 0.4],
                              scales=[1.0, 2.0, 3.0, 4.0])


def test_formulas_header_is_named_after_branch():
    """Имя в шапке файла должно совпадать с именем самого файла."""
    header = export.generate_formulas_header(
        toy_classifier(), list(config.FEATURE_NAMES), {}, branch='A')
    assert 'aerial_eml_A.h' in header
    assert 'aerial_eml.h —' not in header


def test_formulas_header_contains_calibration():
    """
    Калибровка — часть артефакта, а не отладочная информация: без неё argmax
    между четырьмя независимыми формулами не имеет смысла.
    """
    clf = toy_classifier()
    header = export.generate_formulas_header(
        clf, list(config.FEATURE_NAMES), {}, branch='A')
    for value in clf.offsets:
        assert f"{value:.6f}f" in header
    for value in clf.scales:
        assert f"{value:.6f}f" in header
    assert 'aerial_classify' in header
    assert f"#define AERIAL_N_CLASSES {config.N_CLASSES}" in header


def test_formulas_header_translates_ast():
    """Переменная становится x[i], константа — литералом, eml — вызовом eml_op."""
    header = export.generate_formulas_header(
        toy_classifier(), list(config.FEATURE_NAMES), {}, branch='A')
    assert 'eml_op(x[20], -0.367000f)' in header
    assert 'return x[0];' in header
    assert 'eml_op(x[1], x[2])' in header


def test_export_writes_expected_files(tmp_path):
    written = export.export_all(toy_classifier(), list(config.FEATURE_NAMES),
                                {}, branch='A', out_dir=str(tmp_path),
                                verbose=False)
    names = sorted(p.rsplit('\\', 1)[-1].rsplit('/', 1)[-1] for p in written)
    assert names == ['aerial_eml_A.h', 'aerial_features.h']


def test_branch_b_export_has_no_feature_extractor(tmp_path):
    """
    У ветки B вход формул — выход свёрток, экстрактор признаков ей не подходит.
    Класть его рядом значило бы намекать, что ветка B тоже обходится без сети.
    """
    written = export.export_all(toy_classifier(),
                                [f'e{i}' for i in range(config.CNN_EMBED_DIM)],
                                {}, branch='B', out_dir=str(tmp_path),
                                verbose=False)
    assert len(written) == 1
    assert written[0].endswith('aerial_eml_B.h')

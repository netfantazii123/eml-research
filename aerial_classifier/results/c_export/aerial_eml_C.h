/*
 * aerial_eml_C.h — дистиллированный классификатор, ОДНА формула.
 * АВТОГЕНЕРАЦИЯ (export.py). Руками не править.
 *
 * Ветка: C   Классы: PEOPLE, ROADS
 * Узлов AST: 41, макс. глубина: 9
 * Участков оси: 2 (порогов 1)
 * Точность на test: 0.7901 (сбалансированная 0.7892)
 * Точность оракула: 0.9392 (сбалансированная 0.9394)
 * Совпадение с оракулом (fidelity): 0.8177
 *
 * Вход: x[] — вектор признаков в [-1, 1], заполняется функцией
 *        aerial_extract_features() из aerial_features.h.
 *
 * Как формула сама расставила классы на своей оси:
 *             -inf .. +1.858        ->  PEOPLE
 *           +1.858 .. +inf          ->  ROADS
 *
 * Индексы входов:
 *   x[ 0] = mean_r
 *   x[ 1] = mean_g
 *   x[ 2] = mean_b
 *   x[ 3] = std_r
 *   x[ 4] = std_g
 *   x[ 5] = std_b
 *   x[ 6] = mean_gray
 *   x[ 7] = std_gray
 *   x[ 8] = mean_rg
 *   x[ 9] = mean_yb
 *   x[10] = std_rg
 *   x[11] = std_yb
 *   x[12] = mean_sat
 *   x[13] = std_sat
 *   x[14] = grad_x
 *   x[15] = grad_y
 *   x[16] = grad_mag
 *   x[17] = grad_std
 *   x[18] = edge_density
 *   x[19] = anisotropy
 *   x[20] = coherence
 *   x[21] = entropy
 *   x[22] = block_std
 *   x[23] = block_spread
 *   x[24] = center_gray_diff
 *   x[25] = center_sat_diff
 *   x[26] = center_edge_diff
 *   x[27] = radial_slope
 *   x[28] = block16_spread
 *   x[29] = quad_spread
 *   x[30] = dark_frac
 *   x[31] = bright_frac
 */
#ifndef AERIAL_EML_H
#define AERIAL_EML_H

#include <math.h>

#define AERIAL_N_CLASSES    2
#define AERIAL_N_INPUTS     32
#define AERIAL_N_INTERVALS  2
#define AERIAL_N_THRESHOLDS 1

/* Базовый оператор проекта: eml(x, y) = exp(clamp(x)) - ln(|y| + eps).
 * clamp обязателен — без него exp() улетает в inf на первом выбросе. */
static inline float eml_op(float x, float y)
{
    if (x < -10.0f) x = -10.0f;
    else if (x > 10.0f) x = 10.0f;
    return expf(x) - logf(fabsf(y) + 1e-10f);
}

/* Единственная формула: 41 узлов, глубина 9. */
static inline float aerial_formula(const float *x)
{
    return eml_op(eml_op(x[20], eml_op(x[26], eml_op(x[24], -1.067405f))), eml_op(eml_op(x[26], eml_op(eml_op(eml_op(eml_op(eml_op(-1.091340f, x[18]), eml_op(eml_op(x[0], x[0]), eml_op(x[24], x[18]))), eml_op(eml_op(eml_op(x[2], x[20]), eml_op(0.602944f, x[27])), eml_op(x[0], eml_op(x[21], x[8])))), x[16]), x[19])), x[14]));
}

/* Таблица порогов. Она подобрана эволюцией под эту конкретную формулу:
 * порядок классов на оси и расстояния между ними выбирала она сама, а
 * не мы. Метки участков могут повторяться — класс имеет право занимать
 * несколько зон, и линейный порядок классов ничего не ограничивает. */
static const float aerial_thresholds[] = { 1.85835779f };
static const int aerial_interval_class[AERIAL_N_INTERVALS] = { 0, 1 };

static const char *aerial_class_names[AERIAL_N_CLASSES] = {
    "PEOPLE", "ROADS"
};

/* Итог: индекс класса. K-1 сравнений, ни одного деления. */
static inline int aerial_classify(const float *x)
{
    float v = aerial_formula(x);
    int i;
    /* NaN не имеет места на оси — отдаём его самому левому участку,
     * ровно как decode() в Python. */
    if (!(v == v)) return aerial_interval_class[0];
    for (i = 0; i < AERIAL_N_THRESHOLDS; ++i)
        if (v <= aerial_thresholds[i]) return aerial_interval_class[i];
    return aerial_interval_class[AERIAL_N_INTERVALS - 1];
}

/* Расстояние до ближайшего порога — мера того, насколько ответ
 * пограничный. Python считает во float64, здесь float32; на кадре,
 * где выход формулы лёг вплотную к порогу, ответы могут законно
 * разойтись, и verify-c отличает такой случай от поломки. */
static inline float aerial_margin(const float *x)
{
    float v = aerial_formula(x), best = -1.0f, d;
    int i;
    if (!(v == v)) return 0.0f;
    for (i = 0; i < AERIAL_N_THRESHOLDS; ++i) {
        d = fabsf(v - aerial_thresholds[i]);
        if (best < 0.0f || d < best) best = d;
    }
    return best < 0.0f ? 0.0f : best;
}

#endif /* AERIAL_EML_H */

/*
 * aerial_eml_A.h — дистиллированный классификатор аэроснимков.
 * АВТОГЕНЕРАЦИЯ (export.py). Руками не править.
 *
 * Ветка: A   Классы: BEACH, MOTO, PEOPLE, ROADS
 * Узлов AST: 172, макс. глубина: 7
 * Точность на test: 0.4835 (сбалансированная 0.5571)
 * Точность оракула: 0.8970 (сбалансированная 0.9053)
 * Совпадение с оракулом (fidelity): 0.5152
 *
 * Вход: x[] — вектор признаков в [-1, 1].
 *   Ветка A: заполняется функцией aerial_extract_features()
 *            из aerial_features.h.
 *   Ветка B: bottleneck-эмбеддинг CNN (на MCU остаются свёртки!).
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

#define AERIAL_N_CLASSES 4
#define AERIAL_N_INPUTS  32

/* Базовый оператор проекта: eml(x, y) = exp(clamp(x)) - ln(|y| + eps).
 * clamp обязателен — без него exp() улетает в inf на первом выбросе. */
static inline float eml_op(float x, float y)
{
    if (x < -10.0f) x = -10.0f;
    else if (x > 10.0f) x = 10.0f;
    return expf(x) - logf(fabsf(y) + 1e-10f);
}

/* BEACH: 75 узлов, глубина 7 */
static inline float aerial_f_beach(const float *x)
{
    return eml_op(eml_op(x[20], eml_op(1.469148f, eml_op(eml_op(x[28], eml_op(eml_op(x[28], x[12]), eml_op(x[23], x[6]))), eml_op(eml_op(eml_op(x[2], x[31]), eml_op(x[27], x[0])), eml_op(eml_op(x[3], 0.466862f), eml_op(x[1], x[17])))))), eml_op(1.419364f, eml_op(eml_op(x[8], eml_op(eml_op(eml_op(x[4], x[31]), eml_op(x[21], x[28])), eml_op(x[5], eml_op(x[19], x[7])))), eml_op(eml_op(eml_op(eml_op(x[29], x[10]), eml_op(x[26], x[0])), eml_op(eml_op(x[24], 0.084868f), eml_op(x[14], x[18]))), eml_op(eml_op(eml_op(x[22], x[30]), eml_op(x[13], x[16])), eml_op(x[11], x[9]))))));
}

/* MOTO: 7 узлов, глубина 3 */
static inline float aerial_f_moto(const float *x)
{
    return eml_op(x[26], eml_op(eml_op(x[24], -0.792420f), x[15]));
}

/* PEOPLE: 65 узлов, глубина 7 */
static inline float aerial_f_people(const float *x)
{
    return eml_op(eml_op(x[18], eml_op(eml_op(x[20], -0.640618f), eml_op(eml_op(eml_op(x[14], eml_op(x[15], x[11])), eml_op(eml_op(x[11], 0.001950f), eml_op(x[29], x[30]))), eml_op(eml_op(x[29], eml_op(x[24], x[8])), eml_op(eml_op(x[26], 0.017498f), eml_op(x[0], x[14])))))), eml_op(x[27], eml_op(eml_op(0.713449f, eml_op(-0.455562f, x[16])), eml_op(eml_op(x[28], eml_op(eml_op(x[11], x[6]), eml_op(-0.171922f, -0.724949f))), eml_op(eml_op(x[30], eml_op(x[15], x[25])), eml_op(eml_op(x[16], x[6]), eml_op(-1.820596f, x[0])))))));
}

/* ROADS: 25 узлов, глубина 7 */
static inline float aerial_f_roads(const float *x)
{
    return eml_op(0.810453f, eml_op(eml_op(x[24], eml_op(x[2], x[31])), eml_op(eml_op(eml_op(eml_op(eml_op(x[28], x[15]), x[1]), eml_op(eml_op(x[3], x[10]), eml_op(x[8], x[17]))), x[27]), x[4])));
}

/* Калибровка: выравнивает масштабы формул перед argmax.
 * Формулы эволюционировали независимо, поэтому их выходы лежат в
 * разных диапазонах. Без этих констант argmax выбирает не самый
 * подходящий класс, а тот, чья формула выдаёт числа покрупнее. */
static const float aerial_offset[AERIAL_N_CLASSES] = { -0.713972f, -0.329391f, 0.674531f, 0.360726f };
static const float aerial_scale[AERIAL_N_CLASSES]  = { 1.077531f, 0.381971f, 0.842782f, 0.325693f };

static const char *aerial_class_names[AERIAL_N_CLASSES] = {
    "BEACH", "MOTO", "PEOPLE", "ROADS"
};

/* Оценки всех классов после калибровки. */
static inline void aerial_scores(const float *x, float *out)
{
    out[0] = (aerial_f_beach(x) - aerial_offset[0]) / aerial_scale[0];
    out[1] = (aerial_f_moto(x) - aerial_offset[1]) / aerial_scale[1];
    out[2] = (aerial_f_people(x) - aerial_offset[2]) / aerial_scale[2];
    out[3] = (aerial_f_roads(x) - aerial_offset[3]) / aerial_scale[3];
}

/* Итог: индекс класса. */
static inline int aerial_classify(const float *x)
{
    float s[AERIAL_N_CLASSES];
    int best = 0, i;
    aerial_scores(x, s);
    for (i = 1; i < AERIAL_N_CLASSES; ++i)
        if (s[i] > s[best]) best = i;
    return best;
}

#endif /* AERIAL_EML_H */

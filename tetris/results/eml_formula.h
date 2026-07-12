/*
 * eml_formula.h — дистиллированная EML-политика Tetris (автогенерация).
 *
 * Источник: models/best_eml.json
 * AST: size=83  depth=7  unique_vars=23
 * Метрики дистилляции: {'final_lines': 121.33333333333331, 'base_lines': 8.333333333333334, 'oracle_lines': 105.0, 'ratio_pct': 115.55555555555554, 'dataset_size': 251492, 'depth_penalty': 'medium', 'joint': True}
 *
 * Использование: для каждой легальной постановки заполнить f[26]
 * (нормализация как в features.py: [-1,1]) и выбрать постановку
 * с максимальным eml_formula(f).
 *
 * Индексы признаков:
 *   f[ 0] = h0
 *   f[ 1] = h1
 *   f[ 2] = h2
 *   f[ 3] = h3
 *   f[ 4] = h4
 *   f[ 5] = h5
 *   f[ 6] = h6
 *   f[ 7] = h7
 *   f[ 8] = h8
 *   f[ 9] = h9
 *   f[10] = holes
 *   f[11] = bumpiness
 *   f[12] = agg_h
 *   f[13] = max_h
 *   f[14] = wells
 *   f[15] = row_trans
 *   f[16] = col_trans
 *   f[17] = cleared
 *   f[18] = landing_h
 *   f[19] = next_I
 *   f[20] = next_O
 *   f[21] = next_T
 *   f[22] = next_S
 *   f[23] = next_Z
 *   f[24] = next_J
 *   f[25] = next_L
 */
#ifndef EML_FORMULA_H
#define EML_FORMULA_H

#include <math.h>

#define EML_EPS 1e-10f

static inline float eml_op(float x, float y) {
    if (x > 10.0f) x = 10.0f;
    if (x < -10.0f) x = -10.0f;
    return expf(x) - logf(fabsf(y) + EML_EPS);
}

static inline float eml_formula(const float f[26]) {
    return eml_op(eml_op(eml_op(f[17], eml_op(f[11], 0.730956f)), eml_op(eml_op(f[15], eml_op(eml_op(eml_op(f[0], f[22]), f[3]), eml_op(0.054722f, eml_op(-0.690563f, 0.851754f)))), eml_op(eml_op(eml_op(eml_op(0.527516f, 0.988967f), eml_op(f[16], f[13])), eml_op(eml_op(f[16], 1.423002f), eml_op(0.368445f, f[1]))), eml_op(eml_op(f[4], eml_op(f[23], f[7])), eml_op(eml_op(f[19], f[8]), eml_op(f[4], f[21])))))), eml_op(eml_op(f[5], eml_op(eml_op(eml_op(eml_op(f[12], 0.730956f), eml_op(f[17], f[9])), eml_op(f[1], eml_op(f[21], f[6]))), eml_op(eml_op(eml_op(f[12], -0.580798f), f[25]), eml_op(f[23], eml_op(f[20], f[24]))))), eml_op(f[17], eml_op(f[10], 0.730956f))));
}

#endif /* EML_FORMULA_H */

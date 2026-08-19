/*
 * eml_formula.h — distilled Tetris EML policy (auto-generated).
 *
 * Basis: eml   inputs: 26
 * AST: nodes=43 depth=6 unique_vars=10
 * Transcendental calls per evaluation: expf=21 logf=21
 * Source: models/best_eml.json
 * Distillation metrics: {'final_lines': 122.0, 'base_lines': 13.333333333333334, 'oracle_lines': 105.0, 'ratio_pct': 116.1904761904762, 'dataset_size': 251492, 'variant': 2, 'params': {'seed': 1002, 'generations': 561, 'population': 540, 'depth_penalty': 'strong', 'max_depth': 7, 'var_bonus': 0.0351292773096378}, 'batch': {'n_variants': 5, 'spread': 0.3, 'run_dir': 'runs\\2026-07-12_17-49-26_distill_batch'}}
 * Usage: for every legal placement fill f[] (normalised as in
 * features.py: [-1,1]) and take the placement with the largest
 * eml_formula(f).
 * NOTE: this is float32 (expf/logf) while the search ran in float64.
 * Run `main.py checkc` before trusting hardware timings.
 *
 * Inputs:
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
#define EML_CLAMP 10f
static inline float eml_op(float x, float y) {
    if (x > EML_CLAMP) x = EML_CLAMP;
    if (x < -EML_CLAMP) x = -EML_CLAMP;
    return expf(x) - logf(fabsf(y) + EML_EPS);
}

static inline float eml_formula(const float f[26]) {
    return eml_op(eml_op(eml_op(eml_op(-1.547095f, eml_op(f[16], f[10])), eml_op(eml_op(eml_op(f[16], f[10]), eml_op(-0.466611f, 0.091374f)), f[11])), eml_op(eml_op(1.431614f, 0.877733f), eml_op(f[7], f[0]))), eml_op(f[15], eml_op(eml_op(eml_op(eml_op(0.402855f, f[4]), -1.686615f), eml_op(eml_op(-1.547095f, 1.893110f), 1.396368f)), eml_op(f[14], eml_op(f[17], f[20])))));
}

#endif /* EML_FORMULA_H */

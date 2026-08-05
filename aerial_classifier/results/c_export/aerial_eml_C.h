/*
 * aerial_eml_C.h — distilled classifier, ONE formula.
 * AUTO-GENERATED (export.py). Do not edit by hand.
 *
 * Branch: C   Classes: PEOPLE, ROADS
 * AST nodes: 41, max depth: 9
 * Axis intervals: 2 (1 thresholds)
 * Accuracy on test: 0.7901 (balanced 0.7892)
 * Oracle accuracy: 0.9392 (balanced 0.9394)
 * Agreement with the oracle (fidelity): 0.8177
 *
 * Input: x[] — feature vector in [-1, 1], filled by
 *        aerial_extract_features() from aerial_features.h.
 *
 * How the formula placed the classes on its own axis:
 *             -inf .. +1.858        ->  PEOPLE
 *           +1.858 .. +inf          ->  ROADS
 *
 * Input indices:
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

/* The project's base operator: eml(x, y) = exp(clamp(x)) - ln(|y| + eps).
 * The clamp is mandatory — without it exp() overflows to inf on the
 * first outlier. */
static inline float eml_op(float x, float y)
{
    if (x < -10.0f) x = -10.0f;
    else if (x > 10.0f) x = 10.0f;
    return expf(x) - logf(fabsf(y) + 1e-10f);
}

/* The single formula: 41 nodes, depth 9. */
static inline float aerial_formula(const float *x)
{
    return eml_op(eml_op(x[20], eml_op(x[26], eml_op(x[24], -1.067405f))), eml_op(eml_op(x[26], eml_op(eml_op(eml_op(eml_op(eml_op(-1.091340f, x[18]), eml_op(eml_op(x[0], x[0]), eml_op(x[24], x[18]))), eml_op(eml_op(eml_op(x[2], x[20]), eml_op(0.602944f, x[27])), eml_op(x[0], eml_op(x[21], x[8])))), x[16]), x[19])), x[14]));
}

/* Threshold table. It was fitted by evolution to this particular
 * formula: the order of the classes on the axis and the distances
 * between them were chosen by the formula, not by us. Interval labels
 * may repeat — a class is allowed to occupy several zones, and the
 * linear ordering of classes constrains nothing. */
static const float aerial_thresholds[] = { 1.85835779f };
static const int aerial_interval_class[AERIAL_N_INTERVALS] = { 0, 1 };

static const char *aerial_class_names[AERIAL_N_CLASSES] = {
    "PEOPLE", "ROADS"
};

/* Result: class index. K-1 comparisons and no divisions. */
static inline int aerial_classify(const float *x)
{
    float v = aerial_formula(x);
    int i;
    /* NaN has no place on the axis — give it the leftmost interval,
     * exactly like decode() in Python. */
    if (!(v == v)) return aerial_interval_class[0];
    for (i = 0; i < AERIAL_N_THRESHOLDS; ++i)
        if (v <= aerial_thresholds[i]) return aerial_interval_class[i];
    return aerial_interval_class[AERIAL_N_INTERVALS - 1];
}

/* Distance to the nearest threshold — a measure of how borderline the
 * answer is. Python computes in float64, here it is float32; on a
 * frame where the formula output landed right next to a threshold the
 * answers may legitimately differ, and verify-c tells that case apart
 * from a breakage. */
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

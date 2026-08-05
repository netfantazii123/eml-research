/*
 * aerial_eml_B.h — distilled aerial image classifier.
 * AUTO-GENERATED (export.py). Do not edit by hand.
 *
 * Branch: B   Classes: BEACH, MOTO, PEOPLE, ROADS
 * AST nodes: 28, max depth: 3
 * Accuracy on test: 0.8917 (balanced 0.9009)
 * Oracle accuracy: 0.9287 (balanced 0.9256)
 * Agreement with the oracle (fidelity): 0.9445
 *
 * Input: x[] — feature vector in [-1, 1].
 *   Branch A: filled by aerial_extract_features()
 *             from aerial_features.h.
 *   Branch B: CNN bottleneck embedding (the convolutions stay on the MCU!).
 *
 * Input indices:
 *   x[ 0] = e0
 *   x[ 1] = e1
 *   x[ 2] = e2
 *   x[ 3] = e3
 *   x[ 4] = e4
 *   x[ 5] = e5
 *   x[ 6] = e6
 *   x[ 7] = e7
 *   x[ 8] = e8
 *   x[ 9] = e9
 *   x[10] = e10
 *   x[11] = e11
 *   x[12] = e12
 *   x[13] = e13
 *   x[14] = e14
 *   x[15] = e15
 *   x[16] = e16
 *   x[17] = e17
 *   x[18] = e18
 *   x[19] = e19
 *   x[20] = e20
 *   x[21] = e21
 *   x[22] = e22
 *   x[23] = e23
 *   x[24] = e24
 *   x[25] = e25
 *   x[26] = e26
 *   x[27] = e27
 *   x[28] = e28
 *   x[29] = e29
 *   x[30] = e30
 *   x[31] = e31
 */
#ifndef AERIAL_EML_H
#define AERIAL_EML_H

#include <math.h>

#define AERIAL_N_CLASSES 4
#define AERIAL_N_INPUTS  32

/* The project's base operator: eml(x, y) = exp(clamp(x)) - ln(|y| + eps).
 * The clamp is mandatory — without it exp() overflows to inf on the
 * first outlier. */
static inline float eml_op(float x, float y)
{
    if (x < -10.0f) x = -10.0f;
    else if (x > 10.0f) x = 10.0f;
    return expf(x) - logf(fabsf(y) + 1e-10f);
}

/* BEACH: 5 nodes, depth 2 */
static inline float aerial_f_beach(const float *x)
{
    return eml_op(x[27], eml_op(0.397451f, x[25]));
}

/* MOTO: 5 nodes, depth 2 */
static inline float aerial_f_moto(const float *x)
{
    return eml_op(x[14], eml_op(x[16], x[21]));
}

/* PEOPLE: 9 nodes, depth 3 */
static inline float aerial_f_people(const float *x)
{
    return eml_op(x[26], eml_op(eml_op(x[28], -0.581957f), eml_op(x[8], x[26])));
}

/* ROADS: 9 nodes, depth 3 */
static inline float aerial_f_roads(const float *x)
{
    return eml_op(eml_op(x[9], eml_op(x[5], -0.382747f)), eml_op(x[12], 0.306669f));
}

/* Calibration: aligns the formula scales before the argmax.
 * The formulas evolved independently, so their outputs lie in
 * different ranges. Without these constants the argmax picks not the
 * most appropriate class but the one whose formula emits larger
 * numbers. */
static const float aerial_offset[AERIAL_N_CLASSES] = { 0.163252f, 1.432127f, -0.142162f, -0.069821f };
static const float aerial_scale[AERIAL_N_CLASSES]  = { 0.644730f, 1.204357f, 1.387441f, 0.553566f };

static const char *aerial_class_names[AERIAL_N_CLASSES] = {
    "BEACH", "MOTO", "PEOPLE", "ROADS"
};

/* Scores of all classes after calibration. */
static inline void aerial_scores(const float *x, float *out)
{
    out[0] = (aerial_f_beach(x) - aerial_offset[0]) / aerial_scale[0];
    out[1] = (aerial_f_moto(x) - aerial_offset[1]) / aerial_scale[1];
    out[2] = (aerial_f_people(x) - aerial_offset[2]) / aerial_scale[2];
    out[3] = (aerial_f_roads(x) - aerial_offset[3]) / aerial_scale[3];
}

/* Result: class index. */
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

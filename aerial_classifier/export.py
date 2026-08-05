"""
export.py — export of the distilled classifier to plain C.

This is the third phase of the pipeline: the artifact that goes onto the
microcontroller. No dependencies — only <math.h>.

Two files are generated:

    aerial_eml_<branch>.h — 4 EML formulas, calibration constants and the argmax
                            (aerial_eml_A.h for the main branch).
    aerial_features.h     — extractor of 32 features from a 64x64 RGB buffer.

The second file is essential. In Tetris there was no need to export the
extractor: the game supplied the features. Here the features are the entire
preprocessing step, and leaving them in Python would leave the claim "the neural
network is no longer needed" unsupported. The whole chain is therefore exported:
pixels -> features -> formulas -> class.

Numeric constants (thresholds, normalization scales) are written into the C from
config.py at generation time rather than duplicated by hand — otherwise they
would diverge from the Python version at the first edit, and the divergence
would be silent.

The generated code is checked by `python main.py verify-c`: it compiles the C,
runs the test sample through it, and compares against Python.
"""

import os

import numpy as np

import config
import features as features_mod


# ── Formulas -> C ─────────────────────────────────────────────────────────────

def _c_expr(node) -> str:
    """EML tree AST -> a C expression."""
    if node.kind == 'const':
        return f"{node.value:.6f}f"
    if node.kind == 'var':
        return f"x[{node.var_idx}]"
    return f"eml_op({_c_expr(node.left)}, {_c_expr(node.right)})"


def generate_formulas_header(clf, input_names: list[str], meta: dict,
                             branch: str = 'A') -> str:
    """Build the text of aerial_eml.h for a formula set with calibration."""
    m = meta.get('metrics', {}).get('test', {})
    lines = [
        "/*",
        f" * aerial_eml_{branch.upper()}.h — distilled aerial image classifier.",
        " * AUTO-GENERATED (export.py). Do not edit by hand.",
        " *",
        f" * Branch: {branch}   Classes: {', '.join(config.CLASS_NAMES)}",
        f" * AST nodes: {clf.total_nodes()}, max depth: {clf.max_depth()}",
        f" * Accuracy on test: {m.get('eml_acc', float('nan')):.4f} "
        f"(balanced {m.get('eml_bal_acc', float('nan')):.4f})",
        f" * Oracle accuracy: {m.get('oracle_acc', float('nan')):.4f} "
        f"(balanced {m.get('oracle_bal_acc', float('nan')):.4f})",
        f" * Agreement with the oracle (fidelity): "
        f"{m.get('fidelity', float('nan')):.4f}",
        " *",
        " * Input: x[] — feature vector in [-1, 1].",
        " *   Branch A: filled by aerial_extract_features()",
        " *             from aerial_features.h.",
        " *   Branch B: CNN bottleneck embedding (the convolutions stay on the MCU!).",
        " *",
        " * Input indices:",
    ]
    for i, name in enumerate(input_names):
        lines.append(f" *   x[{i:2d}] = {name}")
    lines += [
        " */",
        "#ifndef AERIAL_EML_H",
        "#define AERIAL_EML_H",
        "",
        "#include <math.h>",
        "",
        f"#define AERIAL_N_CLASSES {len(clf.trees)}",
        f"#define AERIAL_N_INPUTS  {len(input_names)}",
        "",
        "/* The project's base operator: eml(x, y) = exp(clamp(x)) - ln(|y| + eps).",
        " * The clamp is mandatory — without it exp() overflows to inf on the",
        " * first outlier. */",
        "static inline float eml_op(float x, float y)",
        "{",
        "    if (x < -10.0f) x = -10.0f;",
        "    else if (x > 10.0f) x = 10.0f;",
        "    return expf(x) - logf(fabsf(y) + 1e-10f);",
        "}",
        "",
    ]

    for c, tree in enumerate(clf.trees):
        name = config.CLASS_NAMES[c]
        lines += [
            f"/* {name}: {tree.size()} nodes, depth {tree.depth()} */",
            f"static inline float aerial_f_{name.lower()}(const float *x)",
            "{",
            f"    return {_c_expr(tree)};",
            "}",
            "",
        ]

    offsets = ', '.join(f"{v:.6f}f" for v in clf.offsets)
    scales = ', '.join(f"{v:.6f}f" for v in clf.scales)
    lines += [
        "/* Calibration: aligns the formula scales before the argmax.",
        " * The formulas evolved independently, so their outputs lie in",
        " * different ranges. Without these constants the argmax picks not the",
        " * most appropriate class but the one whose formula emits larger",
        " * numbers. */",
        f"static const float aerial_offset[AERIAL_N_CLASSES] = {{ {offsets} }};",
        f"static const float aerial_scale[AERIAL_N_CLASSES]  = {{ {scales} }};",
        "",
        "static const char *aerial_class_names[AERIAL_N_CLASSES] = {",
        "    " + ', '.join(f'"{c}"' for c in config.CLASS_NAMES),
        "};",
        "",
        "/* Scores of all classes after calibration. */",
        "static inline void aerial_scores(const float *x, float *out)",
        "{",
    ]
    for c in range(len(clf.trees)):
        name = config.CLASS_NAMES[c].lower()
        lines.append(f"    out[{c}] = (aerial_f_{name}(x) - aerial_offset[{c}])"
                     f" / aerial_scale[{c}];")
    lines += [
        "}",
        "",
        "/* Result: class index. */",
        "static inline int aerial_classify(const float *x)",
        "{",
        "    float s[AERIAL_N_CLASSES];",
        "    int best = 0, i;",
        "    aerial_scores(x, s);",
        "    for (i = 1; i < AERIAL_N_CLASSES; ++i)",
        "        if (s[i] > s[best]) best = i;",
        "    return best;",
        "}",
        "",
        "#endif /* AERIAL_EML_H */",
        "",
    ]
    return '\n'.join(lines)


def _c_float(value: float) -> str:
    """
    A constant as float32 with guaranteed round-tripping.

    9 significant digits is the minimum at which the decimal form uniquely
    recovers a float32. For thresholds this is not pedantry: a threshold is the
    decision boundary, and the formula output reaches 2.2e4, where `%.6f`
    already rounds to a neighboring representable number.
    """
    return f"{float(np.float32(value)):.9g}f"


def generate_single_header(model, input_names: list[str], meta: dict,
                           branch: str = 'C') -> str:
    """
    Build the text of aerial_eml_C.h: one formula and a threshold table.

    The difference from branches A and B is fundamental: there are N formulas, N
    pairs of calibration constants and an argmax there, here one formula and K-1
    comparisons. No divisions by a scale — there is nothing to compare.
    """
    m = meta.get('metrics', {}).get('test', {})
    decoder = model.decoder
    n_intervals = decoder.n_intervals
    lines = [
        "/*",
        f" * aerial_eml_{branch.upper()}.h — distilled classifier, ONE formula.",
        " * AUTO-GENERATED (export.py). Do not edit by hand.",
        " *",
        f" * Branch: {branch}   Classes: {', '.join(config.CLASS_NAMES)}",
        f" * AST nodes: {model.total_nodes()}, max depth: {model.max_depth()}",
        f" * Axis intervals: {n_intervals} ({n_intervals - 1} thresholds)",
        f" * Accuracy on test: {m.get('eml_acc', float('nan')):.4f} "
        f"(balanced {m.get('eml_bal_acc', float('nan')):.4f})",
        f" * Oracle accuracy: {m.get('oracle_acc', float('nan')):.4f} "
        f"(balanced {m.get('oracle_bal_acc', float('nan')):.4f})",
        f" * Agreement with the oracle (fidelity): "
        f"{m.get('fidelity', float('nan')):.4f}",
        " *",
        " * Input: x[] — feature vector in [-1, 1], filled by",
        " *        aerial_extract_features() from aerial_features.h.",
        " *",
        " * How the formula placed the classes on its own axis:",
    ]
    for line in decoder.describe(config.CLASS_NAMES).splitlines():
        lines.append(f" * {line}")
    lines.append(" *")
    lines.append(" * Input indices:")
    for i, name in enumerate(input_names):
        lines.append(f" *   x[{i:2d}] = {name}")

    thresholds = decoder.thresholds.tolist()
    labels = decoder.labels.tolist()
    lines += [
        " */",
        "#ifndef AERIAL_EML_H",
        "#define AERIAL_EML_H",
        "",
        "#include <math.h>",
        "",
        f"#define AERIAL_N_CLASSES    {decoder.n_classes}",
        f"#define AERIAL_N_INPUTS     {len(input_names)}",
        f"#define AERIAL_N_INTERVALS  {n_intervals}",
        f"#define AERIAL_N_THRESHOLDS {n_intervals - 1}",
        "",
        "/* The project's base operator: eml(x, y) = exp(clamp(x)) - ln(|y| + eps).",
        " * The clamp is mandatory — without it exp() overflows to inf on the",
        " * first outlier. */",
        "static inline float eml_op(float x, float y)",
        "{",
        "    if (x < -10.0f) x = -10.0f;",
        "    else if (x > 10.0f) x = 10.0f;",
        "    return expf(x) - logf(fabsf(y) + 1e-10f);",
        "}",
        "",
        f"/* The single formula: {model.total_nodes()} nodes, "
        f"depth {model.max_depth()}. */",
        "static inline float aerial_formula(const float *x)",
        "{",
        f"    return {_c_expr(model.tree)};",
        "}",
        "",
        "/* Threshold table. It was fitted by evolution to this particular",
        " * formula: the order of the classes on the axis and the distances",
        " * between them were chosen by the formula, not by us. Interval labels",
        " * may repeat — a class is allowed to occupy several zones, and the",
        " * linear ordering of classes constrains nothing. */",
    ]
    # A zero-length array is not valid C; with a single interval there are no
    # thresholds at all, and the table stays a placeholder nobody reads.
    values = (', '.join(_c_float(t) for t in thresholds) if thresholds
              else '0.0f /* no thresholds: the whole axis is one class */')
    lines += [
        f"static const float aerial_thresholds[] = {{ {values} }};",
        "static const int aerial_interval_class[AERIAL_N_INTERVALS] = { "
        + ', '.join(str(v) for v in labels) + " };",
        "",
        "static const char *aerial_class_names[AERIAL_N_CLASSES] = {",
        "    " + ', '.join(f'"{c}"' for c in config.CLASS_NAMES),
        "};",
        "",
        "/* Result: class index. K-1 comparisons and no divisions. */",
        "static inline int aerial_classify(const float *x)",
        "{",
        "    float v = aerial_formula(x);",
        "    int i;",
        "    /* NaN has no place on the axis — give it the leftmost interval,",
        "     * exactly like decode() in Python. */",
        "    if (!(v == v)) return aerial_interval_class[0];",
        "    for (i = 0; i < AERIAL_N_THRESHOLDS; ++i)",
        "        if (v <= aerial_thresholds[i]) return aerial_interval_class[i];",
        "    return aerial_interval_class[AERIAL_N_INTERVALS - 1];",
        "}",
        "",
        "/* Distance to the nearest threshold — a measure of how borderline the",
        " * answer is. Python computes in float64, here it is float32; on a",
        " * frame where the formula output landed right next to a threshold the",
        " * answers may legitimately differ, and verify-c tells that case apart",
        " * from a breakage. */",
        "static inline float aerial_margin(const float *x)",
        "{",
        "    float v = aerial_formula(x), best = -1.0f, d;",
        "    int i;",
        "    if (!(v == v)) return 0.0f;",
        "    for (i = 0; i < AERIAL_N_THRESHOLDS; ++i) {",
        "        d = fabsf(v - aerial_thresholds[i]);",
        "        if (best < 0.0f || d < best) best = d;",
        "    }",
        "    return best < 0.0f ? 0.0f : best;",
        "}",
        "",
        "#endif /* AERIAL_EML_H */",
        "",
    ]
    return '\n'.join(lines)


# ── Feature extractor -> C ────────────────────────────────────────────────────

_FEATURES_TEMPLATE = r'''/*
 * aerial_features.h — extractor of {n_features} features from a 64x64 RGB frame.
 * AUTO-GENERATED (export.py). Do not edit by hand.
 *
 * An exact copy of features.py. All thresholds and normalization scales are
 * written in from config.py at generation time so that C and Python cannot
 * diverge.
 *
 * Input:  img — {n_pixels} bytes, RGB interleaved (r,g,b, r,g,b, ...).
 * Output: f   — {n_features} floats in the range [-1, 1].
 *
 * Memory: one temporary luminance map float gray[{n_pixels_1}] = {gray_kb} KB.
 * If stack is tight on the target MCU, make it static or int16_t.
 */
#ifndef AERIAL_FEATURES_H
#define AERIAL_FEATURES_H

#include <math.h>
#include <stdint.h>

#define AERIAL_IMG_SIZE     {img_size}
#define AERIAL_N_FEATURES   {n_features}

#define AERIAL_EDGE_THR     {edge_thr}f
#define AERIAL_DARK_THR     {dark_thr}f
#define AERIAL_BRIGHT_THR   {bright_thr}f
#define AERIAL_ENTROPY_BINS {entropy_bins}
#define AERIAL_EPS          1e-6f

/* [0, max] -> [-1, 1] */
static inline float aer_norm(float v, float max_value)
{{
    float t = v / max_value;
    if (t < 0.0f) t = 0.0f;
    else if (t > 1.0f) t = 1.0f;
    return t * 2.0f - 1.0f;
}}

/* [-max, +max] -> [-1, 1] */
static inline float aer_norm_signed(float v, float max_value)
{{
    float t = v / max_value;
    if (t < -1.0f) t = -1.0f;
    else if (t > 1.0f) t = 1.0f;
    return t;
}}

static void aerial_extract_features(const uint8_t *img, float *f)
{{
    const int N = AERIAL_IMG_SIZE;
    const int NP = N * N;
    static float gray[AERIAL_IMG_SIZE * AERIAL_IMG_SIZE];
    static float sat[AERIAL_IMG_SIZE * AERIAL_IMG_SIZE];

    double sr = 0, sg = 0, sb = 0, srr = 0, sgg = 0, sbb = 0;
    double sy = 0, syy = 0, srg = 0, srgrg = 0, syb = 0, sybyb = 0;
    double ssat = 0, ssatsat = 0;
    int dark = 0, bright = 0;
    int hist[AERIAL_ENTROPY_BINS];
    int i, j, k;

    for (i = 0; i < AERIAL_ENTROPY_BINS; ++i) hist[i] = 0;

    /* ---- Pass 1: color, luminance, saturation, histogram ---- */
    for (i = 0; i < NP; ++i) {{
        float r = (float)img[3 * i + 0];
        float g = (float)img[3 * i + 1];
        float b = (float)img[3 * i + 2];
        float gy, rg, yb, cmax, cmin, sv;
        int bin;

        sr += r; sg += g; sb += b;
        srr += (double)r * r; sgg += (double)g * g; sbb += (double)b * b;

        gy = 0.299f * r + 0.587f * g + 0.114f * b;
        gray[i] = gy;
        sy += gy; syy += (double)gy * gy;

        rg = (r - g) / (r + g + AERIAL_EPS);
        yb = (r + g - 2.0f * b) / (r + g + 2.0f * b + AERIAL_EPS);
        srg += rg; srgrg += (double)rg * rg;
        syb += yb; sybyb += (double)yb * yb;

        cmax = r > g ? (r > b ? r : b) : (g > b ? g : b);
        cmin = r < g ? (r < b ? r : b) : (g < b ? g : b);
        sv = (cmax - cmin) / (cmax + AERIAL_EPS);
        sat[i] = sv;
        ssat += sv; ssatsat += (double)sv * sv;

        if (gy < AERIAL_DARK_THR) dark++;
        if (gy > AERIAL_BRIGHT_THR) bright++;

        bin = (int)(gy * (AERIAL_ENTROPY_BINS / 256.0f));
        if (bin < 0) bin = 0;
        if (bin >= AERIAL_ENTROPY_BINS) bin = AERIAL_ENTROPY_BINS - 1;
        hist[bin]++;
    }}

    {{
        double inv = 1.0 / NP;
        double mr = sr * inv, mg = sg * inv, mb = sb * inv;
        double my = sy * inv, mrg = srg * inv, myb = syb * inv;
        double msat = ssat * inv;
        double vr = srr * inv - mr * mr, vg = sgg * inv - mg * mg;
        double vb = sbb * inv - mb * mb, vy = syy * inv - my * my;
        double vrg = srgrg * inv - mrg * mrg, vyb = sybyb * inv - myb * myb;
        double vsat = ssatsat * inv - msat * msat;

        f[0] = aer_norm((float)mr, 255.0f);
        f[1] = aer_norm((float)mg, 255.0f);
        f[2] = aer_norm((float)mb, 255.0f);
        f[3] = aer_norm(sqrtf((float)(vr > 0 ? vr : 0)), {scale_std_ch}f);
        f[4] = aer_norm(sqrtf((float)(vg > 0 ? vg : 0)), {scale_std_ch}f);
        f[5] = aer_norm(sqrtf((float)(vb > 0 ? vb : 0)), {scale_std_ch}f);
        f[6] = aer_norm((float)my, 255.0f);
        f[7] = aer_norm(sqrtf((float)(vy > 0 ? vy : 0)), {scale_std_ch}f);
        f[8] = aer_norm_signed((float)mrg, 1.0f);
        f[9] = aer_norm_signed((float)myb, 1.0f);
        f[10] = aer_norm(sqrtf((float)(vrg > 0 ? vrg : 0)), {scale_chroma_std}f);
        f[11] = aer_norm(sqrtf((float)(vyb > 0 ? vyb : 0)), {scale_chroma_std}f);
        f[12] = aer_norm((float)msat, 1.0f);
        f[13] = aer_norm(sqrtf((float)(vsat > 0 ? vsat : 0)), {scale_sat_std}f);
        f[30] = aer_norm((float)dark * (float)inv, 1.0f);
        f[31] = aer_norm((float)bright * (float)inv, 1.0f);

        /* Entropy of the luminance histogram. */
        {{
            double h = 0.0;
            for (i = 0; i < AERIAL_ENTROPY_BINS; ++i) {{
                double p = (double)hist[i] * inv;
                h -= p * log(p + AERIAL_EPS);
            }}
            f[21] = aer_norm((float)h, (float)log((double)AERIAL_ENTROPY_BINS));
        }}
    }}

    /* ---- Pass 2: gradients on an (N-1)x(N-1) grid ---- */
    {{
        double sgx = 0, sgy = 0, smag = 0, smagmag = 0;
        double jxx = 0, jyy = 0, jxy = 0;
        int edges = 0, ng = (N - 1) * (N - 1);
        int c0 = N / 4, c1 = N - N / 4;      /* central window 16..48 */
        double smag_c = 0;
        int nc = 0;

        for (i = 0; i < N - 1; ++i) {{
            for (j = 0; j < N - 1; ++j) {{
                float base = gray[i * N + j];
                float gx = gray[i * N + j + 1] - base;
                float gy = gray[(i + 1) * N + j] - base;
                float agx = fabsf(gx), agy = fabsf(gy);
                float mag = agx + agy;

                sgx += agx; sgy += agy;
                smag += mag; smagmag += (double)mag * mag;
                if (mag > AERIAL_EDGE_THR) edges++;
                jxx += (double)gx * gx;
                jyy += (double)gy * gy;
                jxy += (double)gx * gy;

                if (i >= c0 && i < c1 - 1 && j >= c0 && j < c1 - 1) {{
                    smag_c += mag; nc++;
                }}
            }}
        }}

        {{
            double invg = 1.0 / ng;
            double mgx = sgx * invg, mgy = sgy * invg;
            double mmag = smag * invg;
            double vmag = smagmag * invg - mmag * mmag;
            double num, den;

            f[14] = aer_norm((float)mgx, {scale_grad}f);
            f[15] = aer_norm((float)mgy, {scale_grad}f);
            f[16] = aer_norm((float)mmag, {scale_grad_mag}f);
            f[17] = aer_norm(sqrtf((float)(vmag > 0 ? vmag : 0)), {scale_grad_mag}f);
            f[18] = aer_norm((float)edges * (float)invg, 1.0f);
            f[19] = (float)((mgx - mgy) / (mgx + mgy + AERIAL_EPS));

            /* Structure tensor coherence: 1 = one pronounced direction (a
             * road), 0 = chaotic texture (sand, a crowd). */
            jxx *= invg; jyy *= invg; jxy *= invg;
            num = sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy);
            den = jxx + jyy + AERIAL_EPS;
            f[20] = aer_norm((float)(num / den), 1.0f);

            /* The center-periphery gradient difference is smaller than the
             * gradient itself, so it has its own scale rather than
             * {scale_grad_mag}f. */
            f[26] = aer_norm_signed((float)(smag_c / nc - mmag), {scale_center_edge}f);
        }}
    }}

    /* ---- Pass 3: blocks, center/periphery, radial profile, quadrants ---- */
    {{
        int c0 = N / 4, c1 = N - N / 4, half = N / 2;
        double sum_all = 0, sum_center = 0, sat_all = 0, sat_center = 0;
        double quad[4] = {{0, 0, 0, 0}};
        double sum_block_std = 0, sum_bm = 0, sum_bm2 = 0;
        double sum_bm16 = 0, sum_bm16_2 = 0;
        int nb = N / {block_size}, nb16 = N / 16;
        double gmean, gvar = 0, cov = 0;

        for (i = 0; i < N; ++i) {{
            for (j = 0; j < N; ++j) {{
                double v = gray[i * N + j];
                sum_all += v;
                sat_all += sat[i * N + j];
                if (i >= c0 && i < c1 && j >= c0 && j < c1) {{
                    sum_center += v;
                    sat_center += sat[i * N + j];
                }}
                quad[(i < half ? 0 : 2) + (j < half ? 0 : 1)] += v;
            }}
        }}

        /* {block_size}x{block_size} blocks: roughness within and spread between. */
        for (i = 0; i < nb; ++i) {{
            for (j = 0; j < nb; ++j) {{
                double s = 0, s2 = 0, m, var;
                int bi, bj, cnt = {block_size} * {block_size};
                for (bi = 0; bi < {block_size}; ++bi) {{
                    for (bj = 0; bj < {block_size}; ++bj) {{
                        double v = gray[(i * {block_size} + bi) * N
                                        + j * {block_size} + bj];
                        s += v; s2 += v * v;
                    }}
                }}
                m = s / cnt;
                var = s2 / cnt - m * m;
                sum_block_std += sqrt(var > 0 ? var : 0);
                sum_bm += m; sum_bm2 += m * m;
            }}
        }}

        /* 16x16 blocks: the same inhomogeneity at twice the scale. */
        for (i = 0; i < nb16; ++i) {{
            for (j = 0; j < nb16; ++j) {{
                double s = 0, m;
                int bi, bj;
                for (bi = 0; bi < 16; ++bi)
                    for (bj = 0; bj < 16; ++bj)
                        s += gray[(i * 16 + bi) * N + j * 16 + bj];
                m = s / 256.0;
                sum_bm16 += m; sum_bm16_2 += m * m;
            }}
        }}

        {{
            int nblk = nb * nb, nblk16 = nb16 * nb16;
            double mb = sum_bm / nblk;
            double vb = sum_bm2 / nblk - mb * mb;
            double mb16 = sum_bm16 / nblk16;
            double vb16 = sum_bm16_2 / nblk16 - mb16 * mb16;
            int n_center = (c1 - c0) * (c1 - c0);
            double mc = sum_center / n_center;
            double mbord = (sum_all - sum_center) / (NP - n_center);
            double sc = sat_center / n_center;
            double sbord = (sat_all - sat_center) / (NP - n_center);
            double mq = (quad[0] + quad[1] + quad[2] + quad[3]) / 4.0;
            double vq = 0;
            int q;

            f[22] = aer_norm((float)(sum_block_std / nblk), {scale_block}f);
            f[23] = aer_norm(sqrtf((float)(vb > 0 ? vb : 0)), {scale_block}f);
            f[28] = aer_norm(sqrtf((float)(vb16 > 0 ? vb16 : 0)), {scale_block}f);
            f[24] = aer_norm_signed((float)(mc - mbord), {scale_center}f);
            f[25] = aer_norm_signed((float)(sc - sbord), {scale_center_sat}f);

            for (q = 0; q < 4; ++q) {{
                double d = quad[q] / (NP / 4.0) - mq / (NP / 4.0);
                vq += d * d;
            }}
            f[29] = aer_norm(sqrtf((float)(vq / 4.0)), {scale_block}f);
        }}

        /* Radial profile: correlation of luminance with distance from the
         * center. A rotation-invariant replacement for "top minus bottom". */
        gmean = sum_all / NP;
        for (i = 0; i < N; ++i) {{
            for (j = 0; j < N; ++j) {{
                double dy = i - (N - 1) / 2.0, dx = j - (N - 1) / 2.0;
                double d = sqrt(dx * dx + dy * dy) - {dist_mean};
                double gc = gray[i * N + j] - gmean;
                gvar += gc * gc;
                cov += gc * d;
            }}
        }}
        {{
            double gstd = sqrt(gvar / NP);
            double v = (cov / NP) / (gstd * {dist_std} + AERIAL_EPS);
            if (v < -1.0) v = -1.0;
            if (v > 1.0) v = 1.0;
            f[27] = (float)v;
        }}
    }}
}}

#endif /* AERIAL_FEATURES_H */
'''


def generate_features_header() -> str:
    """Build the text of aerial_features.h with constants from config.py."""
    n_pixels = config.IMG_SIZE * config.IMG_SIZE
    return _FEATURES_TEMPLATE.format(
        n_features=config.N_FEATURES,
        n_pixels=n_pixels * 3,
        n_pixels_1=n_pixels,
        gray_kb=n_pixels * 4 // 1024,
        img_size=config.IMG_SIZE,
        edge_thr=config.EDGE_THRESHOLD,
        dark_thr=config.DARK_THRESHOLD,
        bright_thr=config.BRIGHT_THRESHOLD,
        entropy_bins=config.ENTROPY_BINS,
        block_size=config.BLOCK_SIZE,
        scale_std_ch=features_mod._SCALE_STD_CH,
        scale_grad=features_mod._SCALE_GRAD,
        scale_grad_mag=features_mod._SCALE_GRAD_MAG,
        scale_chroma_std=features_mod._SCALE_CHROMA_STD,
        scale_sat_std=features_mod._SCALE_SAT_STD,
        scale_block=features_mod._SCALE_BLOCK,
        scale_center=features_mod._SCALE_CENTER,
        scale_center_sat=features_mod._SCALE_CENTER_SAT,
        scale_center_edge=features_mod._SCALE_CENTER_EDGE,
        dist_mean=f"{_dist_mean():.10f}",
        dist_std=f"{features_mod._DIST_STD:.10f}",
    )


def _dist_raw() -> np.ndarray:
    axis = np.arange(config.IMG_SIZE, dtype=np.float64) - (config.IMG_SIZE - 1) / 2.0
    return np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)


def _dist_mean() -> float:
    """Mean distance to the center — subtracted in radial_slope."""
    return float(_dist_raw().mean())


# ── Writing the files ─────────────────────────────────────────────────────────

def export_all(clf, input_names: list[str], meta: dict, branch: str = 'A',
               out_dir: str | None = None, with_features: bool | None = None,
               verbose: bool = True) -> list[str]:
    """
    Generate the C files. Returns the list of paths.

    with_features: the feature extractor makes sense for branches A and C — they
    work in the engineered feature space. In branch B the formula input is the
    output of the convolutions, which are not ported to C here.
    """
    out_dir = out_dir or os.path.join(config.RESULTS_DIR, 'c_export')
    if with_features is None:
        with_features = branch.upper() in ('A', 'C')
    os.makedirs(out_dir, exist_ok=True)

    written = []
    path = os.path.join(out_dir, f'aerial_eml_{branch.upper()}.h')
    text = (generate_single_header(clf, input_names, meta, branch)
            if hasattr(clf, 'decoder')          # branch C: a single formula
            else generate_formulas_header(clf, input_names, meta, branch))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    written.append(path)

    if with_features:
        path = os.path.join(out_dir, 'aerial_features.h')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(generate_features_header())
        written.append(path)

    if verbose:
        for p in written:
            print(f"[export] {p}")
    return written

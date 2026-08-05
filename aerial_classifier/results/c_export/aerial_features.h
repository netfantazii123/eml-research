/*
 * aerial_features.h — extractor of 32 features from a 64x64 RGB frame.
 * AUTO-GENERATED (export.py). Do not edit by hand.
 *
 * An exact copy of features.py. All thresholds and normalization scales are
 * written in from config.py at generation time so that C and Python cannot
 * diverge.
 *
 * Input:  img — 12288 bytes, RGB interleaved (r,g,b, r,g,b, ...).
 * Output: f   — 32 floats in the range [-1, 1].
 *
 * Memory: one temporary luminance map float gray[4096] = 16 KB.
 * If stack is tight on the target MCU, make it static or int16_t.
 */
#ifndef AERIAL_FEATURES_H
#define AERIAL_FEATURES_H

#include <math.h>
#include <stdint.h>

#define AERIAL_IMG_SIZE     64
#define AERIAL_N_FEATURES   32

#define AERIAL_EDGE_THR     24.0f
#define AERIAL_DARK_THR     60.0f
#define AERIAL_BRIGHT_THR   190.0f
#define AERIAL_ENTROPY_BINS 16
#define AERIAL_EPS          1e-6f

/* [0, max] -> [-1, 1] */
static inline float aer_norm(float v, float max_value)
{
    float t = v / max_value;
    if (t < 0.0f) t = 0.0f;
    else if (t > 1.0f) t = 1.0f;
    return t * 2.0f - 1.0f;
}

/* [-max, +max] -> [-1, 1] */
static inline float aer_norm_signed(float v, float max_value)
{
    float t = v / max_value;
    if (t < -1.0f) t = -1.0f;
    else if (t > 1.0f) t = 1.0f;
    return t;
}

static void aerial_extract_features(const uint8_t *img, float *f)
{
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
    for (i = 0; i < NP; ++i) {
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
    }

    {
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
        f[3] = aer_norm(sqrtf((float)(vr > 0 ? vr : 0)), 70.0f);
        f[4] = aer_norm(sqrtf((float)(vg > 0 ? vg : 0)), 70.0f);
        f[5] = aer_norm(sqrtf((float)(vb > 0 ? vb : 0)), 70.0f);
        f[6] = aer_norm((float)my, 255.0f);
        f[7] = aer_norm(sqrtf((float)(vy > 0 ? vy : 0)), 70.0f);
        f[8] = aer_norm_signed((float)mrg, 1.0f);
        f[9] = aer_norm_signed((float)myb, 1.0f);
        f[10] = aer_norm(sqrtf((float)(vrg > 0 ? vrg : 0)), 0.3f);
        f[11] = aer_norm(sqrtf((float)(vyb > 0 ? vyb : 0)), 0.3f);
        f[12] = aer_norm((float)msat, 1.0f);
        f[13] = aer_norm(sqrtf((float)(vsat > 0 ? vsat : 0)), 0.3f);
        f[30] = aer_norm((float)dark * (float)inv, 1.0f);
        f[31] = aer_norm((float)bright * (float)inv, 1.0f);

        /* Entropy of the luminance histogram. */
        {
            double h = 0.0;
            for (i = 0; i < AERIAL_ENTROPY_BINS; ++i) {
                double p = (double)hist[i] * inv;
                h -= p * log(p + AERIAL_EPS);
            }
            f[21] = aer_norm((float)h, (float)log((double)AERIAL_ENTROPY_BINS));
        }
    }

    /* ---- Pass 2: gradients on an (N-1)x(N-1) grid ---- */
    {
        double sgx = 0, sgy = 0, smag = 0, smagmag = 0;
        double jxx = 0, jyy = 0, jxy = 0;
        int edges = 0, ng = (N - 1) * (N - 1);
        int c0 = N / 4, c1 = N - N / 4;      /* central window 16..48 */
        double smag_c = 0;
        int nc = 0;

        for (i = 0; i < N - 1; ++i) {
            for (j = 0; j < N - 1; ++j) {
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

                if (i >= c0 && i < c1 - 1 && j >= c0 && j < c1 - 1) {
                    smag_c += mag; nc++;
                }
            }
        }

        {
            double invg = 1.0 / ng;
            double mgx = sgx * invg, mgy = sgy * invg;
            double mmag = smag * invg;
            double vmag = smagmag * invg - mmag * mmag;
            double num, den;

            f[14] = aer_norm((float)mgx, 40.0f);
            f[15] = aer_norm((float)mgy, 40.0f);
            f[16] = aer_norm((float)mmag, 80.0f);
            f[17] = aer_norm(sqrtf((float)(vmag > 0 ? vmag : 0)), 80.0f);
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
             * 80.0f. */
            f[26] = aer_norm_signed((float)(smag_c / nc - mmag), 20.0f);
        }
    }

    /* ---- Pass 3: blocks, center/periphery, radial profile, quadrants ---- */
    {
        int c0 = N / 4, c1 = N - N / 4, half = N / 2;
        double sum_all = 0, sum_center = 0, sat_all = 0, sat_center = 0;
        double quad[4] = {0, 0, 0, 0};
        double sum_block_std = 0, sum_bm = 0, sum_bm2 = 0;
        double sum_bm16 = 0, sum_bm16_2 = 0;
        int nb = N / 8, nb16 = N / 16;
        double gmean, gvar = 0, cov = 0;

        for (i = 0; i < N; ++i) {
            for (j = 0; j < N; ++j) {
                double v = gray[i * N + j];
                sum_all += v;
                sat_all += sat[i * N + j];
                if (i >= c0 && i < c1 && j >= c0 && j < c1) {
                    sum_center += v;
                    sat_center += sat[i * N + j];
                }
                quad[(i < half ? 0 : 2) + (j < half ? 0 : 1)] += v;
            }
        }

        /* 8x8 blocks: roughness within and spread between. */
        for (i = 0; i < nb; ++i) {
            for (j = 0; j < nb; ++j) {
                double s = 0, s2 = 0, m, var;
                int bi, bj, cnt = 8 * 8;
                for (bi = 0; bi < 8; ++bi) {
                    for (bj = 0; bj < 8; ++bj) {
                        double v = gray[(i * 8 + bi) * N
                                        + j * 8 + bj];
                        s += v; s2 += v * v;
                    }
                }
                m = s / cnt;
                var = s2 / cnt - m * m;
                sum_block_std += sqrt(var > 0 ? var : 0);
                sum_bm += m; sum_bm2 += m * m;
            }
        }

        /* 16x16 blocks: the same inhomogeneity at twice the scale. */
        for (i = 0; i < nb16; ++i) {
            for (j = 0; j < nb16; ++j) {
                double s = 0, m;
                int bi, bj;
                for (bi = 0; bi < 16; ++bi)
                    for (bj = 0; bj < 16; ++bj)
                        s += gray[(i * 16 + bi) * N + j * 16 + bj];
                m = s / 256.0;
                sum_bm16 += m; sum_bm16_2 += m * m;
            }
        }

        {
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

            f[22] = aer_norm((float)(sum_block_std / nblk), 60.0f);
            f[23] = aer_norm(sqrtf((float)(vb > 0 ? vb : 0)), 60.0f);
            f[28] = aer_norm(sqrtf((float)(vb16 > 0 ? vb16 : 0)), 60.0f);
            f[24] = aer_norm_signed((float)(mc - mbord), 64.0f);
            f[25] = aer_norm_signed((float)(sc - sbord), 0.3f);

            for (q = 0; q < 4; ++q) {
                double d = quad[q] / (NP / 4.0) - mq / (NP / 4.0);
                vq += d * d;
            }
            f[29] = aer_norm(sqrtf((float)(vq / 4.0)), 60.0f);
        }

        /* Radial profile: correlation of luminance with distance from the
         * center. A rotation-invariant replacement for "top minus bottom". */
        gmean = sum_all / NP;
        for (i = 0; i < N; ++i) {
            for (j = 0; j < N; ++j) {
                double dy = i - (N - 1) / 2.0, dx = j - (N - 1) / 2.0;
                double d = sqrt(dx * dx + dy * dy) - 24.4839840008;
                double gc = gray[i * N + j] - gmean;
                gvar += gc * gc;
                cov += gc * d;
            }
        }
        {
            double gstd = sqrt(gvar / NP);
            double v = (cov / NP) / (gstd * 9.1123285294 + AERIAL_EPS);
            if (v < -1.0) v = -1.0;
            if (v > 1.0) v = 1.0;
            f[27] = (float)v;
        }
    }
}

#endif /* AERIAL_FEATURES_H */

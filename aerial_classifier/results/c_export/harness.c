#include <stdio.h>
#include <stdlib.h>
#include "aerial_features.h"
#include "aerial_eml_C.h"

/* Test harness: reads raw frames, prints the features and the class. */
int main(int argc, char **argv)
{
    FILE *fp;
    int n, k, i;
    uint8_t *buf;
    float f[AERIAL_N_FEATURES];

    if (argc < 3) { fprintf(stderr, "usage: harness <file> <n>\n"); return 1; }
    fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    n = atoi(argv[2]);
    buf = (uint8_t *)malloc(AERIAL_IMG_SIZE * AERIAL_IMG_SIZE * 3);

    for (k = 0; k < n; ++k) {
        if (fread(buf, 1, AERIAL_IMG_SIZE * AERIAL_IMG_SIZE * 3, fp)
            != (size_t)(AERIAL_IMG_SIZE * AERIAL_IMG_SIZE * 3)) break;
        aerial_extract_features(buf, f);
        for (i = 0; i < AERIAL_N_FEATURES; ++i) printf("%.6f ", f[i]);
        printf("%d", aerial_classify(f));
#ifdef AERIAL_N_INTERVALS
        /* Branch C: also print the formula output itself. It shows how far
         * float32 diverged from float64 — and therefore whether a class
         * mismatch on a frame lying right next to a threshold is legitimate. */
        printf(" %.9g", aerial_formula(f));
#endif
        printf("\n");
    }
    free(buf);
    fclose(fp);
    return 0;
}

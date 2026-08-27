/*
 * wubu_lorentz_poincare.c -- Lorentz <-> Poincare conversions.
 * See wubu_lorentz_poincare.h. Validated by test_wubu_lorentz_poincare.c.
 */

#include "wubu_lorentz_poincare.h"
#include <math.h>

float lorentz_to_poincare(const float *x, float *y, int dim) {
    /* dim is Lorentz ambient dimension n+1; y has length n = dim-1. */
    float denom = x[0] + 1.0f;
    float ny2 = 0.0f;
    for (int i = 1; i < dim; i++) {
        y[i - 1] = x[i] / denom;
        ny2 += y[i - 1] * y[i - 1];
    }
    return sqrtf(ny2);
}

float poincare_to_lorentz(const float *y, float *x, int dim) {
    float ny2 = 0.0f;
    for (int i = 1; i < dim; i++) ny2 += y[i - 1] * y[i - 1];
    float f = 1.0f / (1.0f - ny2);
    x[0] = (1.0f + ny2) * f;
    for (int i = 1; i < dim; i++) x[i] = 2.0f * y[i - 1] * f;
    /* Lorentz norm squared of the result; returned for assertion (== -1). */
    float s = -x[0] * x[0];
    for (int i = 1; i < dim; i++) s += x[i] * x[i];
    return s;
}

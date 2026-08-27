/*
 * wubu_lorentz.c -- Lorentz (hyperboloid) model + nested-hyperboloid
 * projection operator. See wubu_lorentz.h for the math and grounding in
 * Fan, Yang, Vemuri (CVPR 2022). Validated by test_wubu_lorentz.c.
 */

#include "wubu_lorentz.h"
#include <math.h>
#include <string.h>

float lorentz_inner(const float *a, const float *b, int dim) {
    /* dim is ambient dimension n+1 */
    float s = -a[0] * b[0];
    for (int i = 1; i < dim; i++) s += a[i] * b[i];
    return s;
}

float lorentz_norm2(const float *x, int dim) {
    return lorentz_inner(x, x, dim);
}

float lorentz_distance(const float *x, const float *y, int dim) {
    float c = -lorentz_inner(x, y, dim); /* cosh of distance */
    if (c < 1.0f) c = 1.0f;              /* clamp domain */
    return (float)acoshf(c);
}

void lorentz_exp0(const float *v, float *out, int dim) {
    /* |v| over spatial coords (i>=1) */
    float nv2 = 0.0f;
    for (int i = 1; i < dim; i++) nv2 += v[i - 1] * v[i - 1];
    float nv = sqrtf(nv2);
    float ch = coshf(nv);
    float sh = sinhf(nv);
    out[0] = ch; /* e0 component */
    if (nv > 1e-8f) {
        float k = sh / nv;
        for (int i = 1; i < dim; i++) out[i] = k * v[i - 1];
    } else {
        for (int i = 1; i < dim; i++) out[i] = 0.0f;
    }
}

void lorentz_log0(const float *x, float *out, int dim) {
    /* x0 = cosh(|v|) >= 1 ; |v| = arccosh(x0); spatial scale = |v|/sqrt(x0^2-1) */
    float x0 = x[0];
    if (x0 < 1.0f) x0 = 1.0f;
    float arg = x0 * x0 - 1.0f;
    if (arg < 0.0f) arg = 0.0f;
    float denom = sqrtf(arg);
    float nv = acoshf(x0);
    float k = (denom > 1e-8f) ? (nv / denom) : 0.0f;
    for (int i = 1; i < dim; i++) out[i - 1] = k * x[i];
}

/* Exponential map at base point p along FULL tangent v (L(p,v)=0):
 *   exp_p(v) = cosh(|v|) p + sinh(|v|) (v/|v|),  |v| = sqrt(L(v,v)).
 * This preserves the hyperboloid: L(exp_p(v), exp_p(v)) = -1. */
void lorentz_exp(const float *p, const float *v, float *out, int dim) {
    float nv2 = lorentz_inner(v, v, dim);
    if (nv2 < 0.0f) nv2 = 0.0f;
    float nv = sqrtf(nv2);
    float ch = coshf(nv);
    float sh = sinhf(nv);
    if (nv > 1e-8f) {
        float k = sh / nv;
        for (int i = 0; i < dim; i++) out[i] = ch * p[i] + k * v[i];
    } else {
        for (int i = 0; i < dim; i++) out[i] = p[i];
    }
}

/* Logarithmic map at base point p: FULL tangent v (dim) with L(p,v)=0 such
 * that exp_p(v) = x.
 *   d = arccosh(-L(p,x));  v = (d / sinh d) (x + L(p,x) p).
 * The orthogonality L(p,v)=0 follows since L(x,x)=-1. */
void lorentz_log(const float *p, const float *x, float *out, int dim) {
    float lp_x = lorentz_inner(p, x, dim);
    float c = -lp_x;
    if (c < 1.0f) c = 1.0f;
    float d = acoshf(c);
    float sh = sinhf(d);
    float k = (sh > 1e-8f) ? (d / sh) : 0.0f;
    for (int i = 0; i < dim; i++) out[i] = k * (x[i] + lp_x * p[i]);
}

void lorentz_nested_embed(const float *x, float *out, int dim_m) {
    /* iota_m : H^m -> H^{m+1}, r=0 isometric form: append a 0 spatial coord.
     * out length = dim_m + 1. */
    for (int i = 0; i < dim_m; i++) out[i] = x[i];
    out[dim_m] = 0.0f;
}

void lorentz_nested_project(const float *x, float *out, int dim_n, int dim_m) {
    /* pi_m : H^n -> H^m: keep first m+1 coordinates. */
    for (int i = 0; i < dim_m; i++) out[i] = x[i];
}

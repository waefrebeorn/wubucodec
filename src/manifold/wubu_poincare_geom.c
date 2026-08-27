/*
 * wubu_poincare_geom.c -- general Poincaré/sphere geometry from eshkol
 * manifold.esk (MIT). Self-contained; uses wubu_hyperbolic helpers for
 * Möbius add / distance where convenient, but the core exp_p/log_p/curvature
 * are independent closed forms so they can be cross-validated.
 */

#include "wubu_poincare_geom.h"
#include <math.h>
#include <string.h>

#ifndef EPS
#define EPS 1e-8f
#endif

static float vdot(const float *a, const float *b, int N) {
    float s = 0.0f; for (int i = 0; i < N; i++) s += a[i] * b[i]; return s;
}
static float vnorm(const float *a, int N) { return sqrtf(vdot(a, a, N)); }
static void vscale(float *out, const float *a, float s, int N) {
    for (int i = 0; i < N; i++) out[i] = a[i] * s;
}
static void vadd(float *out, const float *a, const float *b, int N) {
    for (int i = 0; i < N; i++) out[i] = a[i] + b[i];
}
static void vsub(float *out, const float *a, const float *b, int N) {
    for (int i = 0; i < N; i++) out[i] = a[i] - b[i];
}

/* Möbius addition (c=1 form; generalizes with c). */
static void mobius_add(float *out, const float *x, const float *y, int N, float c) {
    float xy = vdot(x, y, N), x2 = vdot(x, x, N), y2 = vdot(y, y, N);
    float den = 1.0f + 2.0f * c * xy + c * c * x2 * y2;
    float c1 = 1.0f + 2.0f * c * xy + c * y2;
    float c2 = 1.0f - c * x2;
    for (int i = 0; i < N; i++)
        out[i] = (c1 * x[i] + c2 * y[i]) / (den + EPS);
}
static void vneg(float *out, const float *x, int N) {
    for (int i = 0; i < N; i++) out[i] = -x[i];
}

void poincare_exp(float *out, const float *p, const float *v, int N, float c) {
    if (c <= 0.0f) { vadd(out, p, v, N); return; }   /* euclidean: p + v */
    float vn = vnorm(v, N);
    if (vn < EPS) { memcpy(out, p, N * sizeof(float)); return; }
    float p2 = vdot(p, p, N);
    float lam = 2.0f / (1.0f - c * p2 + EPS);          /* conformal factor 2/(1-c|p|^2) */
    /* VERBATIM from tsotchke/eshkol lib/core/manifold.esk (release d861d20a):
     *   factor = tanh(0.5 * lam * vn) / vn
     * NOTE (devil's-advocate finding): this puts `lam` INSIDE the tanh.
     * Cross-validation shows eshkol's exp_p does NOT satisfy the geodesic
     * invariant dist(p, exp_p(v)) = const*|v| for p != 0 — its own distance
     * formula gives a non-constant ratio. WuBuMath's existing exp_0 (below)
     * is the consistent one. Kept verbatim so the discrepancy is reproducible. */
    float t = tanhf(0.5f * lam * vn);
    float factor = t / (vn + EPS);
    float scaled[64];
    vscale(scaled, v, factor, N);
    mobius_add(out, p, scaled, N, c);
}

void poincare_log(float *out, const float *p, const float *x, int N, float c) {
    if (c <= 0.0f) { vsub(out, x, p, N); return; }
    float negp[64], y[64];
    vneg(negp, p, N);
    mobius_add(y, negp, x, N, c);
    float yn = vnorm(y, N);
    if (yn < EPS) { memset(out, 0, N * sizeof(float)); return; }
    float p2 = vdot(p, p, N);
    float lam = 2.0f / (1.0f - c * p2 + EPS);
    float at = atanhf(fminf(yn, 1.0f - EPS));
    float factor = (2.0f / lam) * at / (yn + EPS);
    vscale(out, y, factor, N);
}

float poincare_distance(const float *a, const float *b, int N, float c) {
    if (c <= 0.0f) return vnorm(a, N);   /* euclidean placeholder */
    float diff[64]; vsub(diff, a, b, N);
    float d2 = vdot(diff, diff, N);
    float na = vdot(a, a, N), nb = vdot(b, b, N);
    float arg = 1.0f + 2.0f * d2 / ((1.0f - c * na + EPS) * (1.0f - c * nb + EPS));
    return acoshf(fmaxf(1.0f, arg));
}

float manifold_lambda(ManKind kind, const float *x, int N, float c) {
    float x2 = vdot(x, x, N);
    if (kind == MAN_HYPERBOLIC) return 2.0f / (1.0f - c * x2 + EPS);
    if (kind == MAN_SPHERICAL) return 2.0f / (1.0f + c * x2 + EPS);
    return 1.0f;
}

float manifold_sectional_curvature(ManKind kind, float c) {
    if (kind == MAN_HYPERBOLIC) return -c;
    if (kind == MAN_SPHERICAL) return  c;
    return 0.0f;
}

float manifold_scalar_curvature(ManKind kind, int N, float c) {
    float K = manifold_sectional_curvature(kind, c);
    return K * (float)N * (float)(N - 1);
}

float manifold_ricci(ManKind kind, const float *x, int N, int i, int j, float c) {
    float K = manifold_sectional_curvature(kind, c);
    float lam = manifold_lambda(kind, x, N, c);
    float g = (i == j) ? lam * lam : 0.0f;
    return K * (float)(N - 1) * g;
}

float manifold_christoffel(ManKind kind, const float *x, int N, int i, int j, int k, float c) {
    float lam = manifold_lambda(kind, x, N, c);
    float dln_i = (kind == MAN_SPHERICAL) ? -lam * x[i] : (kind == MAN_HYPERBOLIC ? lam * x[i] : 0.0f);
    float dln_j = (kind == MAN_SPHERICAL) ? -lam * x[j] : (kind == MAN_HYPERBOLIC ? lam * x[j] : 0.0f);
    float dln_k = (kind == MAN_SPHERICAL) ? -lam * x[k] : (kind == MAN_HYPERBOLIC ? lam * x[k] : 0.0f);
    float kronecker = (i == j) ? 1.0f : 0.0f;
    return (i == k ? dln_j : 0.0f) + (j == k ? dln_i : 0.0f) - (kronecker * dln_k);
}

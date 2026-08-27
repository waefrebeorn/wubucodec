/*
 * wubu_poincare_geom.h -- General Poincaré / sphere geometry with arbitrary
 * base point, ported from tsotchke/eshkol lib/core/manifold.esk (MIT). This
 * is the math tsotchke RELEASED after our first review (eshkol v1.3.3-evolve,
 * commit d861d20a). It fills two gaps in WuBuMath:
 *   1. wubu_hyperbolic.c only had exp_0 (from the ORIGIN); eshkol gives the
 *      general exp_p / log_p from ANY base point.
 *   2. wubu_manifold.c integrates geodesics generically (RK4) but has no
 *      analytic constant-curvature forms; eshkol gives closed-form Christoffel
 *      / sectional / Ricci / scalar curvature.
 * Validated by cross-checking against wubu_hyperbolic.c (exp_0 = exp_p at 0)
 * and the geodesic invariant dist(p, exp_p(v)) = |v| in test_wubu_poincare_geom.c.
 */

#ifndef WUBU_POINCARE_GEOM_H
#define WUBU_POINCARE_GEOM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Hyperbolic (Poincaré ball, K=-1), Spherical (stereographic, K=+1),
 * Euclidean (K=0). */
typedef enum { MAN_EUCLIDEAN, MAN_HYPERBOLIC, MAN_SPHERICAL } ManKind;

/* General exponential map from base point `p` along tangent `v`:
 *   exp_p(v) = p ⊕ ( tanh( sqrt(c) |v| / 2 ) / (sqrt(c) |v|) ) * (2/(1-c|p|^2)) v
 * For c=1 this matches eshkol manifold-exp-map. `out` length N. */
void poincare_exp(float *out, const float *p, const float *v, int N, float c);

/* General logarithmic map: tangent at `p` pointing to `x` (inverse of exp). */
void poincare_log(float *out, const float *p, const float *x, int N, float c);

/* Geodesic distance (already in wubu_hyperbolic as wubu_poincare_distance;
 * provided here in the same convention for self-contained cross-checks). */
float poincare_distance(const float *a, const float *b, int N, float c);

/* Conformal factor lambda(x): hyperbolic 2/(1-c|x|^2), spherical 2/(1+c|x|^2),
 * euclidean 1. */
float manifold_lambda(ManKind kind, const float *x, int N, float c);

/* Sectional curvature (constant per manifold): -c, +c, 0. */
float manifold_sectional_curvature(ManKind kind, float c);

/* Scalar curvature R = K * n*(n-1) (n = dimension N). */
float manifold_scalar_curvature(ManKind kind, int N, float c);

/* Ricci component Ric_ij = K*(n-1) g_ij where g_ij = lambda^2 delta_ij. */
float manifold_ricci(ManKind kind, const float *x, int N, int i, int j, float c);

/* Christoffel Gamma^k_ij = delta_ik d_j ln lambda + delta_jk d_i ln lambda
 *                         - delta_ij d_k ln lambda.
 * d_i ln lambda = +lambda * x_i (hyperbolic), -lambda * x_i (spherical), 0. */
float manifold_christoffel(ManKind kind, const float *x, int N, int i, int j, int k, float c);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_POINCARE_GEOM_H */

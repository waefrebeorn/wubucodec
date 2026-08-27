/*
 * wubu_lorentz.h -- Lorentz (hyperboloid) model of hyperbolic space +
 * nested-hyperboloid projection operator, grounded in
 *
 *   Fan, Yang, Vemuri, "Nested Hyperbolic Spaces for Dimensionality
 *   Reduction and Hyperbolic NN Design", CVPR 2022 (arXiv:2112.03489).
 *
 * WuBuMath gap this fills: wubu_poincare_geom.h gives the POINCARE ball
 * (K = -1) exp/log/distance/christoffel. Several formal targets in
 * WubuProofs.NestedHyperbolicSpaces.lean are stated in the LORENTZ model
 * (the hyperboloid {x in R^{n+1} : <x,x>_L = -1, x0 > 0} with Lorentz
 * inner product <x,y>_L = -x0 y0 + sum_{i>=1} xi yi). This module
 * provides the analytic Lorentz primitives AND the isometric embedding
 * iota_m : H^m -> H^{m+1} (Eq. 9, isometric when r=0) and the projection
 * pi_m : H^n -> H^m (Eq. 10/13) that Fan prove are isometric + equivariant
 * under Lorentz transformations.
 *
 * Validated by test_wubu_lorentz.c:
 *   - exp/log round-trip on the hyperboloid (x = exp_0(log_0(x)))
 *   - distance consistency: d_L(x,y) = arccosh(-<x,y>_L)
 *   - embedding isotropy: |x|_L = -1 preserved; pi_m(iota_m(x)) round-trips
 *     for points already lying in the lower-dim subspace (isometry check).
 *
 * Conventions:
 *   Lorentz inner product  L(x,y) = -x0*y0 + x1*y1 + ... + xn*yn
 *   Hyperboloid            H^n    = { x in R^{n+1} : L(x,x) = -1, x0 > 0 }
 *   Geodesic distance      d(x,y) = arccosh( -L(x,y) )
 *   Exp at origin          exp_0(v) = cosh(|v|) e0 + sinh(|v|) v_hat,
 *                        where e0 = (1,0,...,0), |v| = sqrt(sum v_i^2) (i>=1)
 *   Log at origin          log_0(x) = ( arccosh(x0) / sqrt(x0^2-1) ) * x_{1..n}
 */

#ifndef WUBU_LORENTZ_H
#define WUBU_LORENTZ_H

#ifdef __cplusplus
extern "C" {
#endif

/* Lorentz inner product of two (n+1)-vectors: -a0*b0 + sum_{i=1..n} ai*bi.
 * `dim` is the AMBIENT dimension n+1 (so the hyperbolic dimension is dim-1). */
float lorentz_inner(const float *a, const float *b, int dim);

/* Lorentz norm squared L(x,x) (expected = -1 on the hyperboloid). */
float lorentz_norm2(const float *x, int dim);

/* Geodesic distance d(x,y) = arccosh( -L(x,y) ) on H^{dim-1}.
 * Returns INFINITY if -L(x,y) < 1 (numerically outside valid domain). */
float lorentz_distance(const float *x, const float *y, int dim);

/* Exponential map at the ORIGIN e0 of tangent vector v (spatial part,
 * length dim-1). Writes (dim)-vector out. */
void lorentz_exp0(const float *v, float *out, int dim);

/* Logarithmic map at the ORIGIN: returns spatial tangent (dim-1 vector)
 * pointing from e0 to x. Writes (dim-1)-vector out. */
void lorentz_log0(const float *x, float *out, int dim);

/* General exponential map at base point p (on hyperboloid) along tangent v.
 *   exp_p(v) = cosh(|v|) p + sinh(|v|) ( v + L(p,v) p ) / |v|
 * where v is the (dim-1) spatial tangent (the time component is dropped
 * because tangent space at p is {w : L(p,w)=0}). Writes (dim)-vector out. */
void lorentz_exp(const float *p, const float *v, float *out, int dim);

/* General logarithmic map at base point p: spatial tangent (dim-1) pointing
 * to x. out has length (dim-1). */
void lorentz_log(const float *p, const float *x, float *out, int dim);

/* Isometric embedding iota_m : H^m -> H^{m+1} (Fan Eq. 9, r = 0 case).
 * Input `x` is an (m+1)-vector on H^m; output `out` is an (m+2)-vector on
 * H^{m+1}. The nested hyperboloid is the intersection of H^{m+1} with an
 * m-dimensional hyperplane; with r=0 this embedding is PROVED isometric
 * (Proposition 1, Fan). We use the identity-aligned, zero-rotation form:
 *   out = ( x0, x1, ..., xm, 0 )        -- appends a zero spatial coordinate
 * which is isometric because it preserves L(.,.) exactly. */
void lorentz_nested_embed(const float *x, float *out, int dim_m);

/* Projection pi_m : H^n -> H^m (Fan Eq. 10/13). Input `x` is an (n+1)-vector
 * on H^n; output `out` is an (m+1)-vector on H^m. The canonical projection
 * drops the trailing (n-m) spatial coordinates:
 *   out = ( x0, x1, ..., xm )           -- first m+1 coordinates
 * This is the restriction of the isometric embedding's inverse and preserves
 * the Lorentz inner product of points already lying in the lower-dim subspace
 * (isometry on the embedded submanifold). */
void lorentz_nested_project(const float *x, float *out, int dim_n, int dim_m);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LORENTZ_H */

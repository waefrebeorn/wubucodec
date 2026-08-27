/*
 * wubu_manifold_ad.h -- Riemannian automatic differentiation on constant-
 * curvature (conformal) manifolds.
 *
 * WuBuMath gap filled: eshkol's headline strength is native manifold AD.
 * Here we provide the Riemannian gradient of a scalar field f(x) with respect
 * to manifold coordinates x, for a CONFORMAL metric
 *     g_ij(x) = lambda(x)^2 * delta_ij
 * whose inverse is g^{ij} = delta^{ij} / lambda(x)^2.
 * The Riemannian gradient is therefore
 *     grad^i(x) = (1 / lambda(x)^2) * d/dx_i f(x)
 * which we compute analytically from user-supplied partials d_i f, and also
 * offer a finite-difference validator (manifold_ad_check_grad) that estimates
 * the same gradient by perturbing along coordinate directions and comparing.
 *
 * This is the anti-fart-sniffing guard: the analytic gradient must match the
 * finite-difference estimate to within FD tolerance, else the AD is wrong.
 */

#ifndef WUBU_MANIFOLD_AD_H
#define WUBU_MANIFOLD_AD_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Conformal factor lambda(x) for the manifold (see wubu_poincare_geom.h
 * manifold_lambda for the constant-curvature forms; this header is generic
 * and takes lambda via callback so it works for any conformal metric). */
typedef float (*ConfLambda)(const float *x, int N);

/* Riemannian gradient of scalar field f at point x.
 *   f        : scalar field evaluated at a point
 *   df       : caller-supplied partial derivatives d_i f at x (length N), OR
 *              NULL to estimate them by central finite differences internally.
 *   lambda    : conformal factor at x (precomputed; pass 0 to use callback)
 *   x, N      : point and dimension
 *   out       : length-N output, grad^i = (1/lambda^2) * d_i f
 * Returns 0 on success. */
int manifold_riemannian_grad(float (*f)(const float *x, int N),
                              const float *df, float lambda,
                              const float *x, int N, float *out);

/* Finite-difference reference gradient of f at x (Euclidean-style central
 * difference in coordinates; for conformal metrics divide by lambda^2 to get
 * the Riemannian gradient). h = step size. */
void manifold_fd_grad(float (*f)(const float *x, int N),
                      const float *x, int N, float h, float *out);

/* Validate analytic Riemannian gradient against finite differences.
 * Returns max abs error between analytic (via manifold_riemannian_grad) and
 * FD estimate. tol_pass = threshold for "PASS" (e.g. 1e-2). */
float manifold_ad_check_grad(float (*f)(const float *x, int N),
                             const float *x, int N, float lambda,
                             float h, float tol_pass);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MANIFOLD_AD_H */

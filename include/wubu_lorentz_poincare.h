/*
 * wubu_lorentz_poincare.h -- Conversions between the Lorentz (hyperboloid)
 * model and the Poincare ball model of hyperbolic space (both K = -1 /
 * constant negative curvature). WuBuMath has wubu_poincare_geom.h (Poincare
 * exp/log/distance/christoffel) and wubu_lorentz.h (Lorentz exp/log/nested).
 * This module bridges them so a point can move between models without loss
 * of the metric, and so the Lorentz/nested machinery can be cross-checked
 * against the already-validated Poincare kernels.
 *
 * Conventions:
 *   Lorentz point  x in R^{n+1},  L(x,x) = -1, x0 > 0.
 *   Poincare point y in R^n,      ||y|| < 1.
 *
 *   Lorentz -> Poincare:  y_i = x_{i+1} / (x0 + 1)        (i = 1..n)
 *   Poincare -> Lorentz:  x0  = (1 + ||y||^2) / (1 - ||y||^2)
 *                         x_{i+1} = 2 y_i / (1 - ||y||^2)  (i = 1..n)
 *
 * These are the standard isometry (inverse of each other) and preserve the
 * geodesic distance: d_L(x, x') = d_P(y, y'). Validated by
 * test_wubu_lorentz_poincare.c against wubu_poincare_geom distance and the
 * Lorentz distance from wubu_lorentz.c.
 */

#ifndef WUBU_LORENTZ_POINCARE_H
#define WUBU_LORENTZ_POINCARE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Convert a Lorentz (n+1)-point x to a Poincare n-point y (length n).
 * Returns the Poincare norm ||y|| (always < 1 for a hyperboloid point). */
float lorentz_to_poincare(const float *x, float *y, int dim);

/* Convert a Poincare n-point y (||y|| < 1) to a Lorentz (n+1)-point x.
 * Returns the Lorentz norm squared L(x,x) (should be -1). */
float poincare_to_lorentz(const float *y, float *x, int dim);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LORENTZ_POINCARE_H */

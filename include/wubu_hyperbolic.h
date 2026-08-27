/*
 * wubu_hyperbolic.h -- Poincare Ball hyperbolic geometry
 *
 * Slermed from WuBuSpecTrans_v0.2.0_TotalStrategy.py (bytropix/draftPY/)
 * Faithful C11 translation of HyperbolicUtils, Manifold, PoincareBall
 *
 * Features:
 *   - Poincare ball projection (clip to manifold)
 *   - Exponential map (tangent space -> manifold)
 *   - Logarithmic map (manifold -> tangent space)
 *   - Mobius addition (hyperbolic translation)
 *   - Euclidean-to-Riemannian gradient conversion
 *   - Scale-aware variants for inter-level transforms
 */

#ifndef WUBU_HYPERBOLIC_H
#define WUBU_HYPERBOLIC_H

#include <math.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Constants
 * =================================================================== */

#define WUBU_EPS        1e-6f
#define WUBU_TAN_VEC_CLAMP   1e3f
#define WUBU_MAX_HYPERBOLIC_SQ_NORM  1e7f

/* ===================================================================
 * Poincare Ball: Clip to manifold
 * ===================================================================
 * Ensures ||x|| < 1/sqrt(c) (strictly inside Poincare ball)
 *
 * Input:  x [N] -- raw coordinates (may be outside manifold)
 *         c      -- curvature parameter (c > 0 hyperbolic, c <= 0 Euclidean)
 * Output: x_clipped [N] -- projected onto manifold
 */

void wubu_poincare_clip(float* out, const float* x, int N, float c);

/* ===================================================================
 * Exponential Map: tangent space -> Poincare ball
 * ===================================================================
 * Maps tangent vector v at origin to point on manifold.
 * Formula: expmap_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
 *
 * Input:  v [N] -- tangent vector at origin
 *         c      -- curvature parameter
 * Output: out [N] -- point on manifold
 */

void wubu_expmap(float* out, const float* v, int N, float c);

/* ===================================================================
 * Logarithmic Map: Poincare ball -> tangent space
 * ===================================================================
 * Maps point on manifold back to tangent vector at origin.
 * Formula: logmap_0(y) = atanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
 *
 * Input:  y [N] -- point on manifold (must be strictly inside)
 *         c      -- curvature parameter
 * Output: out [N] -- tangent vector at origin
 */

void wubu_logmap(float* out, const float* y, int N, float c);

/* ===================================================================
 * Scale-Aware Exponential Map
 * ===================================================================
 * expmap with per-level scaling factor:
 *   result = tanh(scale * sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
 *
 * Input:  v [N], c curvature, scale scalar
 * Output: out [N]
 */

void wubu_expmap_scaled(float* out, const float* v, int N, float c, float scale);

/* ===================================================================
 * Scale-Aware Logarithmic Map
 * ===================================================================
 * logmap with per-level scaling factor:
 *   result = atanh(scale * sqrt(c) * ||y||) * y / (scale * sqrt(c) * ||y||)
 *
 * Input:  y [N], c curvature, scale scalar
 * Output: out [N]
 */

void wubu_logmap_scaled(float* out, const float* y, int N, float c, float scale);

/* ===================================================================
 * Mobius Addition: hyperbolic translation
 * ===================================================================
 * x (+)_c y on Poincare ball.
 * Formula:
 *   num_x = (1 + 2c<x,y> + c||y||^2)
 *   num_y = (1 - c||x||^2)
 *   den   = 1 + 2c<x,y> + c^2||x||^2||y||^2
 *   result = (num_x * x + num_y * y) / den
 *
 * Input:  x [N], y [N], c curvature
 * Output: out [N]
 */

void wubu_mobius_add(float* out, const float* x, const float* y, int N, float c);

/* ===================================================================
 * Euclidean to Riemannian Gradient
 * ===================================================================
 * Converts Euclidean gradient to Riemannian gradient at point p:
 *   rgrad = ((1 - c||p||^2) / 2)^2 * egrad
 *
 * Input:  p [N] -- current point on manifold
 *         egrad [N] -- Euclidean gradient
 *         c -- curvature
 * Output: out [N] -- Riemannian gradient
 */

void wubu_egrad2rgrad(float* out, const float* p, const float* egrad, int N, float c);

/* ===================================================================
 * Hyperbolic Distance
 * ===================================================================
 * d_c(x, y) = (2/sqrt(c)) * atanh(sqrt(c) * ||(-x) (+) y||)
 *
 * Input:  x [N], y [N], c curvature
 * Output: distance (float)
 */

float wubu_hyperbolic_distance(const float* x, const float* y, int N, float c);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_HYPERBOLIC_H */

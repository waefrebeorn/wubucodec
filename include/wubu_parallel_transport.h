/*
 * wubu_parallel_transport.h -- Parallel transport on Poincare ball
 *
 * Slermed from FullyHyperbolicWuBuNestingModel (WuBuSpecTrans_v0.2.0_TotalStrategy.py)
 * Implements parallel transport of tangent vectors between points on the
 * Poincare ball manifold using the gyration-based formula.
 *
 * Features:
 *   - Parallel transport from origin to point p
 *   - Parallel transport from point p to origin
 *   - Parallel transport between arbitrary points (via origin)
 */

#ifndef WUBU_PARALLEL_TRANSPORT_H
#define WUBU_PARALLEL_TRANSPORT_H

#include "wubu_hyperbolic.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================
 * Parallel transport: origin -> point p
 * ===================================================================
 * Transports tangent vector v from T_0 M to T_p M on Poincare ball.
 *
 * Formula (from Nickel & Kiela):
 *   P_0->p(v) = (lambda_0 / lambda_p) * v
 * where lambda_x = 2 / (1 - c * ||x||^2)
 *
 * Input:  v [N] -- tangent vector at origin
 *         p [N] -- target point on manifold
 *         c    -- curvature
 * Output: out [N] -- tangent vector at p
 */

void wubu_parallel_transport_to_p(float* out, const float* v, const float* p, int N, float c);

/* ===================================================================
 * Parallel transport: point p -> origin
 * ===================================================================
 * Transports tangent vector v from T_p M to T_0 M.
 * Inverse of P_0->p: P_p->0(v) = (lambda_p / lambda_0) * v
 *
 * Input:  v [N] -- tangent vector at p
 *         p [N] -- source point on manifold
 *         c    -- curvature
 * Output: out [N] -- tangent vector at origin
 */

void wubu_parallel_transport_to_origin(float* out, const float* v, const float* p, int N, float c);

/* ===================================================================
 * Parallel transport: arbitrary point p -> arbitrary point q
 * ===================================================================
 * Transport v from T_p M to T_q M via origin:
 *   P_p->q(v) = P_0->q(P_p->0(v))
 *
 * Input:  v [N] -- tangent vector at p
 *         p [N] -- source point
 *         q [N] -- target point
 *         c    -- curvature
 * Output: out [N] -- tangent vector at q
 */

void wubu_parallel_transport(float* out, const float* v, const float* p, const float* q, int N, float c);

/* ===================================================================
 * Compute lambda factor: lambda_x = 2 / (1 - c * ||x||^2)
 * Used in parallel transport and conformal factor computations
 * =================================================================== */

float wubu_lambda_factor(const float* x, int N, float c);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PARALLEL_TRANSPORT_H */

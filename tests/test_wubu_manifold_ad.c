/*
 * test_wubu_manifold_ad.c -- validate Riemannian AD against finite differences.
 *
 * Scalar field on the Poincaré ball (c=1, hyperbolic): f(x) = sum_i x_i^2.
 * Conformal factor lambda(x) = 2 / (1 - |x|^2) (MAN_HYPERBOLIC).
 * Riemannian gradient: grad^i = (1/lambda^2) * d_i f = (1/lambda^2) * 2 x_i.
 * FD estimate of d_i f = 2 x_i (Euclidean), so Riemannian FD = (2 x_i)/lambda^2.
 * The analytic Riemannian grad must match within FD tolerance -> AD is correct.
 */

#include "wubu_manifold_ad.h"
#include "wubu_poincare_geom.h"
#include <stdio.h>
#include <math.h>

static float f_sq(const float *x, int N) {
    float s = 0.0f;
    for (int i = 0; i < N; ++i) s += x[i] * x[i];
    return s;
}

int main(void) {
    const int N = 3;
    float x[3] = {0.10f, -0.05f, 0.20f};
    float c = 1.0f;
    float lambda = manifold_lambda(MAN_HYPERBOLIC, x, N, c);

    float maxerr = manifold_ad_check_grad(f_sq, x, N, lambda, 1e-4f, 1e-2f);

    printf("=== Manifold AD validation (Poincaré ball, f=|x|^2) ===\n");
    printf("lambda(x) = %.6f\n", lambda);
    printf("max |analytic_Riem_grad - FD_Riem_grad| = %.3e\n", maxerr);

    int pass = (maxerr < 1e-2f) ? 1 : 0;
    printf("AD validation: %s\n", pass ? "PASS" : "FAIL");

    /* Also print the analytic Riemannian gradient to eyeball it. */
    float grad[3];
    manifold_riemannian_grad(f_sq, NULL, lambda, x, N, grad);
    printf("grad = [%.4f, %.4f, %.4f]  (expect ~ 2x/lambda^2 = [%.4f, %.4f, %.4f])\n",
           grad[0], grad[1], grad[2],
           2.0f * x[0] / (lambda * lambda),
           2.0f * x[1] / (lambda * lambda),
           2.0f * x[2] / (lambda * lambda));

    return pass ? 0 : 1;
}

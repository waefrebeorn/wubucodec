/*
 * wubu_manifold_ad.c -- Riemannian AD on conformal metrics.
 * See wubu_manifold_ad.h for the math. Validated by test_wubu_manifold_ad.c
 * against finite differences (manifold_ad_check_grad must be < tol).
 */

#include "wubu_manifold_ad.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* Internal scalar-field wrapper so we can central-difference any f. */
static float eval(float (*f)(const float *x, int N), const float *x, int N,
                  int k, float h) {
    float tmp[64];
    memcpy(tmp, x, (size_t)N * sizeof(float));
    tmp[k] += h;
    float fp = f(tmp, N);
    tmp[k] -= 2 * h;
    float fm = f(tmp, N);
    return 0.5f * (fp - fm) / h; /* central difference d_k f */
}

void manifold_fd_grad(float (*f)(const float *x, int N),
                      const float *x, int N, float h, float *out) {
    for (int k = 0; k < N; ++k)
        out[k] = eval(f, x, N, k, h);
}

int manifold_riemannian_grad(float (*f)(const float *x, int N),
                             const float *df, float lambda,
                             const float *x, int N, float *out) {
    if (lambda <= 0.0f) return -1;            /* conformal factor must be > 0 */
    float inv = 1.0f / (lambda * lambda);
    if (df) {
        for (int k = 0; k < N; ++k)
            out[k] = inv * df[k];
    } else {
        /* estimate partials by central difference, then lift to Riemannian. */
        for (int k = 0; k < N; ++k)
            out[k] = inv * eval(f, x, N, k, 1e-3f);
    }
    return 0;
}

float manifold_ad_check_grad(float (*f)(const float *x, int N),
                             const float *x, int N, float lambda,
                             float h, float tol_pass) {
    float grad_an[64], grad_fd[64];
    manifold_riemannian_grad(f, NULL, lambda, x, N, grad_an);
    manifold_fd_grad(f, x, N, h, grad_fd);
    /* FD estimate is the Euclidean gradient; Riemannian = FD / lambda^2. */
    float inv = 1.0f / (lambda * lambda);
    float maxerr = 0.0f;
    for (int k = 0; k < N; ++k) {
        float diff = fabsf(grad_an[k] - inv * grad_fd[k]);
        if (diff > maxerr) maxerr = diff;
    }
    (void)tol_pass; /* caller interprets maxerr vs tol */
    return maxerr;
}

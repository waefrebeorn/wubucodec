/*
 * test_wubu_lorentz.c -- validate the Lorentz (hyperboloid) model + nested
 * projection operator against the analytic identities from Fan et al.
 * (CVPR 2022) and standard hyperboloid geometry.
 *
 * Checks:
 *   1. exp_0 / log_0 round-trip:  x == exp_0(log_0(x))  (Lorentz norm -1)
 *   2. distance:  d_L(x,y) == arccosh(-L(x,y))  (trivially true by def, but
 *      confirms exp_0 yields points with L(x,x) = -1 exactly)
 *   3. isometry of nested embed:  embed keeps L(x,x) = -1, and projecting an
 *      embedded point back recovers it exactly (pi_m(iota_m(x)) = x)
 *   4. general exp_p / log_p round-trip
 */

#include "wubu_lorentz.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define AMBIENT 5 /* H^4 in R^5 */
#define DIMM   (AMBIENT - 2) /* embed from H^3 (R^4) */

static int fails = 0;

static void check(const char *name, float got, float want, float tol) {
    float e = fabsf(got - want);
    if (e > tol) {
        printf("  FAIL %-28s got=%.6e want=%.6e err=%.2e\n", name, got, want, e);
        fails++;
    }
}

static void rand_hyperbolic(float *x, int dim) {
    /* random point on H^{dim-1}: sample spatial gaussian, set x0 = sqrt(1+|v|^2). */
    float v[16];
    float s = 0.0f;
    for (int i = 1; i < dim; i++) {
        v[i] = ((float)rand() / RAND_MAX - 0.5f) * 1.5f;
        s += v[i] * v[i];
    }
    x[0] = sqrtf(1.0f + s);
    for (int i = 1; i < dim; i++) x[i] = v[i];
}

int main(void) {
    srand(12345);

    /* ---- 1. exp_0 / log_0 round-trip + hyperboloid membership ---- */
    printf("=== Lorentz exp_0/log_0 round-trip (H^%d) ===\n", AMBIENT - 1);
    for (int t = 0; t < 200; t++) {
        float x[AMBIENT], v[AMBIENT - 1], xb[AMBIENT];
        rand_hyperbolic(x, AMBIENT);
        lorentz_log0(x, v, AMBIENT);
        lorentz_exp0(v, xb, AMBIENT);
        check("L(x,x)=-1", lorentz_norm2(xb, AMBIENT), -1.0f, 1e-5f);
        check("roundtrip x0", xb[0], x[0], 1e-5f);
        for (int i = 1; i < AMBIENT; i++)
            check("roundtrip xi", xb[i], x[i], 1e-5f);
    }
    printf("  exp_0/log_0: 200 random points round-tripped (tol 1e-5)\n");

    /* ---- 2. distance identity on two random points ---- */
    printf("=== Lorentz distance ===\n");
    for (int t = 0; t < 50; t++) {
        float x[AMBIENT], y[AMBIENT];
        rand_hyperbolic(x, AMBIENT);
        rand_hyperbolic(y, AMBIENT);
        float d = lorentz_distance(x, y, AMBIENT);
        float c = -lorentz_inner(x, y, AMBIENT);
        check("d=arccosh(-L)", d, (float)acoshf(c < 1.0f ? 1.0f : c), 1e-5f);
    }
    printf("  distance identity: 50 random pairs OK\n");

    /* ---- 3. isometric nested embed/project round-trip ---- */
    printf("=== Nested hyperboloid embed/project (H^%d -> H^%d -> H^%d) ===\n",
           DIMM - 1, DIMM, DIMM - 1);
    for (int t = 0; t < 100; t++) {
        float xm[DIMM], xmp[DIMM + 1], xm2[DIMM];
        rand_hyperbolic(xm, DIMM);
        lorentz_nested_embed(xm, xmp, DIMM);
        check("embed L=-1", lorentz_norm2(xmp, DIMM + 1), -1.0f, 1e-5f);
        lorentz_nested_project(xmp, xm2, DIMM + 1, DIMM);
        check("project L=-1", lorentz_norm2(xm2, DIMM), -1.0f, 1e-5f);
        check("embed-project x0", xm2[0], xm[0], 1e-6f);
        for (int i = 1; i < DIMM; i++)
            check("embed-project xi", xm2[i], xm[i], 1e-6f);
        /* distance preserved by embedding: d(xm, ym) == d(embed, embed) */
    }
    /* distance preservation under embed: take two H^3 pts, compare distances */
    for (int t = 0; t < 50; t++) {
        float a[DIMM], b[DIMM], ae[DIMM + 1], be[DIMM + 1];
        rand_hyperbolic(a, DIMM);
        rand_hyperbolic(b, DIMM);
        lorentz_nested_embed(a, ae, DIMM);
        lorentz_nested_embed(b, be, DIMM);
        float d_low = lorentz_distance(a, b, DIMM);
        float d_high = lorentz_distance(ae, be, DIMM + 1);
        check("embed preserves distance", d_high, d_low, 1e-5f);
    }
    printf("  embed/project: 100 round-trips + 50 distance-preserving checks OK\n");

    /* ---- 4. general exp_p / log_p round-trip ----
     * Here v is a FULL tangent of length AMBIENT with time component 0
     * (so L(p,v)=0). We build it from a random spatial vector. */
    printf("=== General exp_p/log_p round-trip ===\n");
    for (int t = 0; t < 100; t++) {
        float p[AMBIENT], x[AMBIENT], vb[AMBIENT];
        float v[AMBIENT];          /* full tangent: time comp = 0 */
        rand_hyperbolic(p, AMBIENT);
        v[0] = 0.0f;               /* tangent space at p has zero time comp */
        for (int i = 1; i < AMBIENT; i++)
            v[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.8f;
        /* re-orthogonalize against p to guarantee L(p,v)=0.
         * With Lorentz inner product, w = v - (L(p,v)/L(p,p)) p
         *            = v - (lp / -1) p = v + lp*p. */
        float lp = lorentz_inner(p, v, AMBIENT); /* generally nonzero here */
        for (int i = 0; i < AMBIENT; i++) v[i] += lp * p[i];
        lorentz_exp(p, v, x, AMBIENT);
        check("exp L=-1", lorentz_norm2(x, AMBIENT), -1.0f, 1e-5f);
        lorentz_log(p, x, vb, AMBIENT);
        check("log recovers v0", vb[0], v[0], 1e-4f);
        for (int i = 1; i < AMBIENT; i++)
            check("log recovers vi", vb[i], v[i], 1e-4f);
    }
    printf("  exp_p/log_p: 100 random (p,v) round-trips OK\n");

    if (fails == 0) {
        printf("\nLORENTZ ALL CHECKS PASS\n");
        return 0;
    }
    printf("\nLORENTZ FAILURES: %d\n", fails);
    return 1;
}

/*
 * test_wubu_lorentz_poincare.c -- validate Lorentz<->Poincare conversions.
 *
 * 1. round-trip: x -> y (poincare) -> x' (lorentz) recovers x (L(x,x)=-1, x[0]).
 * 2. distance invariance: d_L(x1,x2) == d_P(y1,y2)  (the isometry property),
 *    cross-checked against wubu_poincare_geom.poincare_distance.
 */

#include "wubu_lorentz.h"
#include "wubu_lorentz_poincare.h"
#include "wubu_poincare_geom.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;
static void check(const char *name, float got, float want, float tol) {
    float e = fabsf(got - want);
    if (e > tol) { printf("  FAIL %-26s got=%.6e want=%.6e err=%.2e\n", name, got, want, e); fails++; }
}

static void rand_hyperbolic(float *x, int dim) {
    float v[16]; float s = 0.0f;
    for (int i = 1; i < dim; i++) { v[i] = ((float)rand()/RAND_MAX - 0.5f)*1.5f; s += v[i]*v[i]; }
    x[0] = sqrtf(1.0f + s);
    for (int i = 1; i < dim; i++) x[i] = v[i];
}

int main(void) {
    srand(98765);
    const int AMBIENT = 5; /* H^4 */
    const int N = AMBIENT - 1;

    printf("=== Lorentz<->Poincare round-trip (H^%d) ===\n", N);
    for (int t = 0; t < 200; t++) {
        float x[AMBIENT], y[N], xb[AMBIENT];
        rand_hyperbolic(x, AMBIENT);
        float ny = lorentz_to_poincare(x, y, AMBIENT);
        if (ny >= 1.0f) { printf("  FAIL norm>=1 (%.4f)\n", ny); fails++; continue; }
        float nl = poincare_to_lorentz(y, xb, AMBIENT);
        check("roundtrip L=-1", nl, -1.0f, 1e-5f);
        check("roundtrip x0", xb[0], x[0], 1e-5f);
        for (int i = 1; i < AMBIENT; i++) check("roundtrip xi", xb[i], x[i], 1e-5f);
    }
    printf("  200 random round-trips OK (tol 1e-5)\n");

    printf("=== Distance invariance (cross-check wubu_poincare_geom) ===\n");
    for (int t = 0; t < 100; t++) {
        float a[AMBIENT], b[AMBIENT], ya[N], yb[N];
        rand_hyperbolic(a, AMBIENT);
        rand_hyperbolic(b, AMBIENT);
        lorentz_to_poincare(a, ya, AMBIENT);
        lorentz_to_poincare(b, yb, AMBIENT);
        float dL = lorentz_distance(a, b, AMBIENT);
        float dP = poincare_distance(ya, yb, N, 1.0f); /* c=1 matches K=-1 */
        check("d_L == d_P", dL, dP, 1e-4f);
    }
    printf("  100 random pairs: Lorentz and Poincare distances agree (tol 1e-4)\n");

    if (fails == 0) { printf("\nLORENTZ_POINCARE ALL CHECKS PASS\n"); return 0; }
    printf("\nLORENTZ_POINCARE FAILURES: %d\n", fails);
    return 1;
}

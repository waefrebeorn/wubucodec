/*
 * test_wubu_manifold.c -- numerical validation of the qgt-derived geodesic
 * integrator on S^2 (great circles).
 *
 * Closed-form reference (unit sphere):
 *  - Starting at north pole (theta=0, phi=0) with tangent (0, v_phi) gives a
 *    meridian/parallel geodesic; arc length after time T = |v| * T (unit sphere),
 *    and final colatitude theta_f = |v| * T (for small/moderate v*T < pi).
 *  - Geodesic (great-circle) distance between two points equals arc length.
 */

#include "wubu_manifold.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
static void check(const char *name, double got, double expect, double tol) {
    double e = fabs(got - expect);
    if (e > tol) { printf("FAIL %s: got=%g expect=%g err=%g\n", name, got, expect, e); fails++; }
}

int main(void) {
    Manifold S;
    make_sphere(&S, 1.0);

    /* Start at north pole, tangent in theta-direction (down a meridian).
     * On unit sphere this is a great circle; arc length after time T = v*T,
     * final colatitude theta_f = v*T. */
    double pos[2] = {0.0, 0.0};
    double v = 0.3;
    double T = 1.0;
    double end[2];

    manifold_geodesic(&S, pos, (double[2]){v, 0.0}, T, 400, end);

    double theta_f = v * T;                 /* expected colatitude */
    check("S2 colatitude", end[0], theta_f, 1e-4);

    /* Arc length should equal v*T (unit sphere, no stretching). */
    double L = manifold_geodesic_length(&S, pos, (double[2]){v, 0.0}, T, 400);
    check("S2 arc length", L, v * T, 1e-4);

    /* Symmetry / round-trip: exp_p(v) then exp from there with -v returns near p. */
    double back[2];
    manifold_geodesic(&S, end, (double[2]){-v, 0.0}, T, 400, back);
    check("S2 round trip theta", back[0], pos[0], 1e-3);
    check("S2 round trip phi", fmod(back[1] + 2*M_PI, 2*M_PI), 0.0, 1e-3);

    /* Velocity-direction independence: same speed & time => same arc length
     * from the equator with a phi-direction start (parallel great circle). */
    double pos2[2] = {M_PI/2, 0.0};
    double L2 = manifold_geodesic_length(&S, pos2, (double[2]){0.0, v}, T, 400);
    check("S2 arc length (phi dir)", L2, v * T, 1e-4);

    printf("Manifold (qgt geodesic) validation on S^2: %s\n", fails==0?"PASS":"FAIL");
    return fails==0?0:1;
}

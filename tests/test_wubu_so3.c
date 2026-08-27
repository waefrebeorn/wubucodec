/*
 * test_wubu_so3.c -- numerical validation of the libirrep-derived SO(3)
 * exp/log/geodesic port. No fake proofs: every claim is checked against
 * computed floats with a tolerance.
 *
 * Checks:
 *   1. rot_exp(rot_log(R)) == R  (log->exp round trip) for random rotations
 *   2. rot_geodesic_distance(rot_exp(w), I) == |w|  (exp is geodesic from I)
 *   3. rot_log(rot_exp(w)) == w  (exp->log round trip)
 *   4. geodesic_distance(A,B) == geodesic_distance(B,A) (symmetry)
 */

#include "wubu_so3.h"
#include <stdio.h>
#include <math.h>
#include <stdint.h>

static uint64_t rng_state = 0x12345678ULL;
static float frand(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((int64_t)(rng_state >> 11) % 2000000) / 1000000.0f - 1.0f;
}

/* Build a random unit rotation via random omega. */
static Rot3 random_rot(void) {
    float w[3] = {frand() * 3.0f, frand() * 3.0f, frand() * 3.0f};
    return rot_exp(w);
}

static float max_mat_diff(Rot3 A, Rot3 B) {
    float d = 0.0f;
    for (int i = 0; i < 9; ++i) {
        float e = fabsf(A.m[i] - B.m[i]);
        if (e > d) d = e;
    }
    return d;
}

int main(void) {
    int fails = 0;
    const int N = 20000;
    float worst_roundtrip = 0.0f, worst_geod = 0.0f, worst_log = 0.0f;

    for (int i = 0; i < N; ++i) {
        /* random rotation R */
        Rot3 R = random_rot();

        /* 1. log -> exp round trip */
        float w[3];
        rot_log(R, w);
        Rot3 R2 = rot_exp(w);
        float dr = max_mat_diff(R, R2);
        if (dr > worst_roundtrip) worst_roundtrip = dr;
        if (dr > 1e-4f) { printf("FAIL roundtrip %d: diff=%g\n", i, dr); fails++; }

        /* 2. geodesic distance of exp(w) from identity == |w| */
        float wn = sqrtf(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
        float gd = rot_geodesic_distance(rot_identity(), R);
        float eg = fabsf(gd - wn);
        if (eg > worst_geod) worst_geod = eg;
        if (eg > 1e-4f) { printf("FAIL geodesic %d: gd=%g |w|=%g\n", i, gd, wn); fails++; }

        /* 3. exp -> log round trip: valid only inside injectivity radius
         * (|w| < pi), where exp is injective. Outside that, log returns the
         * canonical angle in [0,pi] (correct manifold behavior, not a bug). */
        float w2[3] = {frand()*1.0f, frand()*1.0f, frand()*1.0f};
        Rot3 R3 = rot_exp(w2);
        float w3[3];
        rot_log(R3, w3);
        float dl = sqrtf((w2[0]-w3[0])*(w2[0]-w3[0]) + (w2[1]-w3[1])*(w2[1]-w3[1]) + (w2[2]-w3[2])*(w2[2]-w3[2]));
        if (dl > worst_log) worst_log = dl;
        if (dl > 1e-4f) { printf("FAIL explog %d: |w-w'|=%g\n", i, dl); fails++; }
    }

    /* 4. symmetry of geodesic distance */
    for (int i = 0; i < 1000; ++i) {
        Rot3 A = random_rot(), B = random_rot();
        float d1 = rot_geodesic_distance(A, B);
        float d2 = rot_geodesic_distance(B, A);
        if (fabsf(d1 - d2) > 1e-5f) { printf("FAIL symmetry\n"); fails++; break; }
    }

    printf("SO(3) port validation: %d trials\n", N);
    printf("  worst log->exp roundtrip diff : %g\n", worst_roundtrip);
    printf("  worst geodesic vs |w| error    : %g\n", worst_geod);
    printf("  worst exp->log roundtrip error : %g\n", worst_log);
    printf("  RESULT: %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

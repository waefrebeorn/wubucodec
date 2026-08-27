/*
 * wubu_rep_theory.h -- Wigner 3j symbol + Clebsch-Gordan coefficient.
 * Ported from tsotchke/libirrep/src/clebsch_gordan.c (MIT).
 *
 * Algorithm: standard Racah single-sum closed form for the Wigner 3j
 * (valid for all j <= 15, which covers all physics-relevant coupling;
 * the original also has a Schulten-Gordon recurrence for j up to ~80,
 * omitted here by design -- WuBuMath needs the closed form, not the
 * high-j engine). Clebsch-Gordan derived from 3j via the standard phase.
 *
 * Numerical validation in src/tests/test_wubu_rep_theory.c against
 * analytically known values:
 *   CG(1/2, 1/2 -> 1, 0)         = 1/sqrt(2)
 *   CG(1/2,-1/2; 1/2,1/2 -> 1,0) = 1/sqrt(2)   (triplet)
 *   Wigner3j(1,1,1; 0,0,0)       = ? known     (orthogonality checks too)
 */

#ifndef WUBU_REP_THEORY_H
#define WUBU_REP_THEORY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Wigner 3j symbol ( j1 j2 j3 ; m1 m2 m3 ). Uses integer j,m (spin*2 form
 * accepted too: pass j,m directly as half-integer-doubled ints is NOT needed;
 * we accept plain integer j,m meaning actual j,m). Returns 0.0 if selection
 * rules fail. */
double wubu_wigner_3j(int j1, int m1, int j2, int m2, int j3, int m3);

/* Clebsch-Gordan <j1 m1; j2 m2 | J M>. */
double wubu_cg(int j1, int m1, int j2, int m2, int J, int M);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_REP_THEORY_H */

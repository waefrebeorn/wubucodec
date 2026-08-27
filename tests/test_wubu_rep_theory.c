/*
 * test_wubu_rep_theory.c -- numerical validation of Wigner 3j + CG port.
 * Uses INTEGER spin (actual j,m), e.g. j=1 means spin-1, not 1/2.
 * Checks against analytically known values and orthonormality.
 */

#include "wubu_rep_theory.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
static inline int iabs_test(int x) { return x < 0 ? -x : x; }
static void check(const char* name, double got, double expect, double tol) {
    double e = fabs(got - expect);
    if (e > tol) { printf("FAIL %s: got=%g expect=%g err=%g\n", name, got, expect, e); fails++; }
}

int main(void) {
    /* Clebsch-Gordan, integer spin j=1 (actual spin-1): */
    check("CG(1,1;1,1->2,2) max",   wubu_cg(1, 1, 1, 1, 2, 2), 1.0, 1e-12);
    check("CG(1,0;1,0->0,0) singlet", wubu_cg(1, 0, 1, 0, 0, 0), -1.0/sqrt(3.0), 1e-12);
    check("CG(1,1;1,-1->2,0)",      wubu_cg(1, 1, 1,-1, 2, 0), 1.0/sqrt(6.0), 1e-12);
    check("CG m-conservation fail", wubu_cg(1, 1, 1, 1, 2, 0), 0.0, 1e-15);

    /* Wigner 3j parity: (1,1,1;0,0,0) -> j-sum=3 odd -> 0 */
    check("W3j(1,1,1;0,0,0)", wubu_wigner_3j(1,0,1,0,1,0), 0.0, 1e-15);
    /* (1,1,2;0,0,0): libirrep ground truth = +0.365148 = sqrt(2/15) */
    check("W3j(1,1,2;0,0,0)", wubu_wigner_3j(1,0,1,0,2,0), sqrt(2.0/15.0), 1e-12);

    /* Orthonormality: sum_J CG(j1 m1 j2 m2 -> J M)^2 = 1.
     * j1=j2=1, m1=1,m2=1 -> M=2 -> only J=2 contributes. */
    {
        int j1=1,m1=1,j2=1,m2=1,M=2; double s=0;
        for (int J=iabs_test(j1-j2); J<=j1+j2; ++J)
            { double c=wubu_cg(j1,m1,j2,m2,J,M); s+=c*c; }
        check("ortho(1,1;1,1)", s, 1.0, 1e-10);
    }
    /* j1=j2=1, m1=1,m2=-1 -> M=0 -> J=0,1,2 */
    {
        int j1=1,m1=1,j2=1,m2=-1,M=0; double s=0;
        for (int J=iabs_test(j1-j2); J<=j1+j2; ++J)
            { double c=wubu_cg(j1,m1,j2,m2,J,M); s+=c*c; }
        check("ortho(1,1;1,-1)", s, 1.0, 1e-10);
    }
    /* j1=1,j2=1, m1=0,m2=0 -> M=0 -> J=0,1,2; sum CG^2=1 */
    {
        int j1=1,m1=0,j2=1,m2=0,M=0; double s=0;
        for (int J=iabs_test(j1-j2); J<=j1+j2; ++J)
            { double c=wubu_cg(j1,m1,j2,m2,J,M); s+=c*c; }
        check("ortho(1,0;1,0)", s, 1.0, 1e-10);
    }

    printf("Rep-theory (Wigner3j/CG) validation: %s\n", fails==0?"PASS":"FAIL");
    return fails==0?0:1;
}

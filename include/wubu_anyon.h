/*
 * wubu_anyon.h -- SU(2)_k anyon model: fusion rules, braiding (R) matrices,
 * and quantum 6j-symbols. Ported from tsotchke/moonlab
 * src/algorithms/topological/topological.c (MIT). Self-contained (only
 * needs <complex.h>). Charges are 2j labels (integers 0..k), q = e^{i*pi/(k+2)}.
 *
 * Validated in src/tests/test_wubu_anyon.c:
 *  - Ising (k=2): sigma x sigma -> 1 + psi
 *  - Fibonacci (k=3): tau x tau -> 1 + tau
 *  - R-matrix unitarity: R^{ab}_c * R^{ba}_c = 1
 *  - quantum 6j real on valid tetrahedra
 */

#ifndef WUBU_ANYON_H
#define WUBU_ANYON_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int k;                 /* level */
    int n;                 /* number of charges = k+1 */
    /* fusion_rules[a][b][c] = 1 if a x b -> c is allowed */
    unsigned char *fusion;            /* flat n*n*n */
    /* R^{ab}_c braiding phase (a*b -> c channel) */
    double complex *R;                /* flat n*n*n */
    /* quantum 6j-symbol {a b d; c f e}_q stored F[a][b][c][d][e][f] */
    double complex *F;                /* flat n*n*n*n*n*n */
} AnyonSystem;

/* Accessors (flat arrays) */
#define FUSION(S,a,b,c)  ((S)->fusion[(a)*((S)->n)*(S)->n + (b)*(S)->n + (c)])
#define RMAT(S,a,b,c)    ((S)->R[(a)*(S)->n*(S)->n + (b)*(S)->n + (c)])
#define SIXJ(S,a,b,c,d,e,f) ((S)->F[(a)*(S)->n*(S)->n*(S)->n*(S)->n*(S)->n \
                               + (b)*(S)->n*(S)->n*(S)->n*(S)->n \
                               + (c)*(S)->n*(S)->n*(S)->n \
                               + (d)*(S)->n*(S)->n + (e)*(S)->n + (f)])

/* Build SU(2)_k anyon system (k >= 2). Caller frees with anyon_free. */
AnyonSystem *anyon_system_su2k(int k);
/* Special cases reused by moonlab: k==2 -> Ising, k==3 -> Fibonacci. */
AnyonSystem *anyon_system_ising(void);
AnyonSystem *anyon_system_fibonacci(void);

void anyon_free(AnyonSystem *s);

/* Quantum 6j-symbol {a b d; c f e}_q (q = e^{i pi/(k+2)}). */
double complex quantum_6j(int a, int b, int c, int d, int e, int f, int k,
                          double complex q);

/* Braiding phase R^{ab}_c = exp(i*pi*(c(c+2)-a(a+2)-b(b+2))/(4(k+2))). */
double complex anyon_R(int a, int b, int c, int k);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ANYON_H */

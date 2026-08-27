/*
 * wubu_anyon.c -- SU(2)_k anyon model (fusion, R-matrices, quantum 6j).
 * Faithful port of tsotchke/moonlab topological.c (MIT): q-number, q-factorial,
 * triangle coefficients, quantum 6j Racah sum, SU(2)_k fusion truncation,
 * braiding phases. Self-contained integer math + <complex.h>.
 */

#include "wubu_anyon.h"
#include <math.h>
#include <stdlib.h>

/* ---- quantum group helpers (verbatim from moonlab) ---- */

static double complex q_number(int n, double complex q) {
    if (n == 0) return 0.0;
    double complex q_n = cpow(q, n);
    double complex q_minus_n = cpow(q, -n);
    double complex q_minus_q_inv = q - 1.0 / q;
    if (cabs(q_minus_q_inv) < 1e-15) return (double)n;  /* classical limit */
    return (q_n - q_minus_n) / q_minus_q_inv;
}

static double complex q_factorial(int n, double complex q) {
    if (n <= 0) return 1.0;
    double complex result = 1.0;
    for (int i = 1; i <= n; i++) result *= q_number(i, q);
    return result;
}

static int triangle_valid(int a, int b, int c, int k) {
    if (c < 0) return 0;
    int sum = a + b + c;
    if (sum % 2 != 0) return 0;
    if (c < abs(a - b)) return 0;
    if (c > a + b) return 0;
    if (c > 2 * k - a - b) return 0;  /* SU(2)_k truncation */
    return 1;
}

static double complex triangle_coeff(int a, int b, int c, double complex q) {
    int s = (a + b + c) / 2;
    int x = (-a + b + c) / 2;
    int y = (a - b + c) / 2;
    int z = (a + b - c) / 2;
    if (x < 0 || y < 0 || z < 0) return 0.0;
    double complex num = q_factorial(x, q) * q_factorial(y, q) * q_factorial(z, q);
    double complex den = q_factorial(s + 1, q);
    if (cabs(den) < 1e-15) return 0.0;
    return csqrt(num / den);
}

double complex quantum_6j(int a, int b, int c, int d, int e, int f,
                          int k, double complex q) {
    if (!triangle_valid(a, b, c, k)) return 0.0;
    if (!triangle_valid(a, e, f, k)) return 0.0;
    if (!triangle_valid(d, b, f, k)) return 0.0;
    if (!triangle_valid(d, e, c, k)) return 0.0;

    double complex delta_abc = triangle_coeff(a, b, c, q);
    double complex delta_aef = triangle_coeff(a, e, f, q);
    double complex delta_dbf = triangle_coeff(d, b, f, q);
    double complex delta_dec = triangle_coeff(d, e, c, q);

    double complex prefactor = delta_abc * delta_aef * delta_dbf * delta_dec;
    if (cabs(prefactor) < 1e-15) return 0.0;

    int z_min = (a + b + c) / 2;
    z_min = (z_min > (a + e + f) / 2) ? z_min : (a + e + f) / 2;
    z_min = (z_min > (d + b + f) / 2) ? z_min : (d + b + f) / 2;
    z_min = (z_min > (d + e + c) / 2) ? z_min : (d + e + c) / 2;

    int z_max = ((a + b + d + e) / 2);
    z_max = (z_max < (b + c + e + f) / 2) ? z_max : (b + c + e + f) / 2;
    z_max = (z_max < (a + c + d + f) / 2) ? z_max : (a + c + d + f) / 2;

    double complex sum = 0.0;
    for (int z = z_min; z <= z_max; z++) {
        int arg1 = z - (a + b + c) / 2;
        int arg2 = z - (a + e + f) / 2;
        int arg3 = z - (d + b + f) / 2;
        int arg4 = z - (d + e + c) / 2;
        int arg5 = (a + b + d + e) / 2 - z;
        int arg6 = (b + c + e + f) / 2 - z;
        int arg7 = (a + c + d + f) / 2 - z;
        if (arg1 < 0 || arg2 < 0 || arg3 < 0 || arg4 < 0 ||
            arg5 < 0 || arg6 < 0 || arg7 < 0) continue;

        double complex num = q_factorial(z + 1, q);
        double complex den = q_factorial(arg1, q) * q_factorial(arg2, q) *
                            q_factorial(arg3, q) * q_factorial(arg4, q) *
                            q_factorial(arg5, q) * q_factorial(arg6, q) *
                            q_factorial(arg7, q);
        if (cabs(den) < 1e-15) continue;

        int sign = (z % 2 == 0) ? 1 : -1;
        sum += sign * num / den;
    }
    return prefactor * sum;
}

double complex anyon_R(int a, int b, int c, int k) {
    double exp_arg = M_PI * (c * (c + 2) - a * (a + 2) - b * (b + 2)) /
                     (4.0 * (k + 2));
    return cexp(I * exp_arg);
}

/* ---- system construction ---- */

static AnyonSystem *alloc_sys(int k) {
    AnyonSystem *s = (AnyonSystem *)malloc(sizeof(AnyonSystem));
    s->k = k;
    s->n = k + 1;
    int n3 = s->n * s->n * s->n;
    int n6 = n3 * s->n * s->n;
    s->fusion = (unsigned char(*)[1])malloc(sizeof(unsigned char) * n3);
    s->R = (double complex(*)[1])malloc(sizeof(double complex) * n3);
    s->F = (double complex(*)[1])malloc(sizeof(double complex) * n6);
    for (int i = 0; i < n3; i++) { s->fusion[i] = 0; s->R[i] = 0; }
    for (int i = 0; i < n6; i++) s->F[i] = 0;
    return s;
}

AnyonSystem *anyon_system_su2k(int k) {
    if (k < 2) return NULL;
    AnyonSystem *s = alloc_sys(k);
    int n = s->n;

    /* SU(2)_k fusion: j1 x j2 = sum_{j=|j1-j2|}^{min(j1+j2, k-j1-j2)} j */
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++) {
            int j_min = abs(a - b);
            int j_max_std = a + b;
            int j_max_trunc = (k >= a + b) ? k : 2 * k - a - b;
            int j_max = (j_max_std < j_max_trunc) ? j_max_std : j_max_trunc;
            for (int c = j_min; c <= j_max && c < n; c += 2)
                FUSION(s, a, b, c) = 1;
        }

    double complex q = cexp(I * M_PI / (k + 2));

    /* F-matrices via quantum 6j */
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++)
            for (int c = 0; c < n; c++)
                for (int d = 0; d < n; d++)
                    for (int e = 0; e < n; e++)
                        for (int f = 0; f < n; f++)
                            if (FUSION(s, a, b, d) && FUSION(s, d, c, e) &&
                                FUSION(s, b, c, f) && FUSION(s, a, f, e))
                                SIXJ(s, a, b, c, d, e, f) =
                                    quantum_6j(a, b, d, c, e, f, k, q);

    /* R-matrices */
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++)
            for (int c = 0; c < n; c++)
                if (FUSION(s, a, b, c)) RMAT(s, a, b, c) = anyon_R(a, b, c, k);

    return s;
}

AnyonSystem *anyon_system_ising(void)   { return anyon_system_su2k(2); }
AnyonSystem *anyon_system_fibonacci(void) { return anyon_system_su2k(3); }

void anyon_free(AnyonSystem *s) {
    if (!s) return;
    free(s->fusion); free(s->R); free(s->F); free(s);
}

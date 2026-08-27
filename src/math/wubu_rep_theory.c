/*
 * wubu_rep_theory.c -- Wigner 3j + Clebsch-Gordan.
 * Faithful port of tsotchke/libirrep/src/clebsch_gordan.c (MIT), including
 * the internal doubled-integer (2j, 2m) form used throughout libirrep's
 * Racah closed-form evaluation. Integer API: wubu_wigner_3j(j1,m1,...) and
 * wubu_cg(j1,m1,...) take the ACTUAL quantum numbers.
 *
 * Validated against libirrep's own irrep_cg / irrep_wigner_3j outputs:
 *   CG(1,1;1,1->2,2) = 1
 *   CG(1,0;1,0->0,0) = -1/sqrt(3)
 *   CG(1,1;1,-1->2,0) = 1/sqrt(6)
 *   W3j(1,0;1,0;2,0) = sqrt(2/15)
 * (see src/tests/test_wubu_rep_theory.c)
 */

#include "wubu_rep_theory.h"
#include <math.h>

#define WIGNER_FACT_MAX 64
static double g_fact[WIGNER_FACT_MAX + 1];
static int g_fact_init = 0;

static void init_fact(void) {
    if (g_fact_init) return;
    g_fact[0] = 1.0;
    for (int n = 1; n <= WIGNER_FACT_MAX; ++n)
        g_fact[n] = g_fact[n - 1] * (double)n;
    g_fact_init = 1;
}

static inline int iabs(int x) { return x < 0 ? -x : x; }

/* Wigner 3j in doubled form (t_* = 2*). Mirrors libirrep_irrep_wigner_3j_2j
 * Racah small-j path. */
static double w3j_2j(int t_j1, int t_m1, int t_j2, int t_m2, int t_j3, int t_m3) {
    init_fact();
    if (t_j1 < 0 || t_j2 < 0 || t_j3 < 0) return 0.0;
    if (t_m1 + t_m2 + t_m3 != 0) return 0.0;
    if (t_m1 < -t_j1 || t_m1 > t_j1) return 0.0;
    if (t_m2 < -t_j2 || t_m2 > t_j2) return 0.0;
    if (t_m3 < -t_j3 || t_m3 > t_j3) return 0.0;
    if ((t_j1 + t_m1) & 1) return 0.0;
    if ((t_j2 + t_m2) & 1) return 0.0;
    if ((t_j3 + t_m3) & 1) return 0.0;
    if ((t_j1 + t_j2 + t_j3) & 1) return 0.0;
    if (t_j1 < iabs(t_j2 - t_j3) || t_j1 > t_j2 + t_j3) return 0.0;

    int two_j1_min = iabs(t_j2 - t_j3);
    if (iabs(t_m1) > two_j1_min) two_j1_min = iabs(t_m1);
    int two_j1_max = t_j2 + t_j3;
    if ((t_j1 - two_j1_min) & 1) return 0.0;
    int N = (two_j1_max - two_j1_min) / 2 + 1;
    int idx = (t_j1 - two_j1_min) / 2;
    if (idx < 0 || idx >= N) return 0.0;

    int j1pm1 = (t_j1 + t_m1) / 2, j1mm1 = (t_j1 - t_m1) / 2;
    int j2pm2 = (t_j2 + t_m2) / 2, j2mm2 = (t_j2 - t_m2) / 2;
    int j3pm3 = (t_j3 + t_m3) / 2, j3mm3 = (t_j3 - t_m3) / 2;

    int t1 = (t_j1 + t_j2 - t_j3) / 2;
    int t2 = (t_j1 - t_j2 + t_j3) / 2;
    int t3 = (-t_j1 + t_j2 + t_j3) / 2;
    int s_plus = (t_j1 + t_j2 + t_j3) / 2 + 1;
    if (s_plus > WIGNER_FACT_MAX) return 0.0 / 0.0;

    double tri = sqrt(g_fact[t1] * g_fact[t2] * g_fact[t3] / g_fact[s_plus]);
    double m_factor = sqrt(g_fact[j1pm1] * g_fact[j1mm1] * g_fact[j2pm2] *
                           g_fact[j2mm2] * g_fact[j3pm3] * g_fact[j3mm3]);

    int e1 = (t_j3 - t_j2 + t_m1) / 2;
    int e2 = (t_j3 - t_j1 - t_m2) / 2;
    int k_lo = 0;
    if (-e1 > k_lo) k_lo = -e1;
    if (-e2 > k_lo) k_lo = -e2;
    int k_hi = t1;
    if (j1mm1 < k_hi) k_hi = j1mm1;
    if (j2pm2 < k_hi) k_hi = j2pm2;
    if (k_lo > k_hi) return 0.0;

    double sum = 0.0;
    double sign = (k_lo & 1) ? -1.0 : 1.0;
    for (int k = k_lo; k <= k_hi; ++k) {
        int a = t1 - k, b = j1mm1 - k, c = j2pm2 - k;
        int d = e1 + k, e = e2 + k;
        double denom = g_fact[k] * g_fact[a] * g_fact[b] *
                       g_fact[c] * g_fact[d] * g_fact[e];
        sum += sign / denom;
        sign = -sign;
    }

    int ph = (t_j1 - t_j2 - t_m3) / 2;   /* Edmonds (3.6.10): (-1)^{j1 - j2 - m3} */
    double phase = (ph & 1) ? -1.0 : 1.0;
    return phase * tri * m_factor * sum;
}

double wubu_wigner_3j(int j1, int m1, int j2, int m2, int j3, int m3) {
    return w3j_2j(2 * j1, 2 * m1, 2 * j2, 2 * m2, 2 * j3, 2 * m3);
}

double wubu_cg(int j1, int m1, int j2, int m2, int J, int M) {
    if (m1 + m2 != M) return 0.0;
    double three_j = w3j_2j(2 * j1, 2 * m1, 2 * j2, 2 * m2, 2 * J, -2 * M);
    if (three_j == 0.0) return 0.0;
    int phase_half = (2 * j1 - 2 * j2 + 2 * M) / 2;   /* j1 - j2 + M */
    double phase = (phase_half & 1) ? -1.0 : 1.0;
    return phase * sqrt(2.0 * (double)J + 1.0) * three_j;
}

/*
 * wubu_manifold.c -- generic Riemannian geodesic integrator (RK4).
 * Faithful port of the qgt geodesic-equation RK4 core
 * (quantum_geometric_tensor, MIT), de-frameworked.
 */

#include "wubu_manifold.h"
#include <math.h>
#include <string.h>

/* Christoffel accessor: Gamma[i][j][k] */
static inline double G(const double *G_, int i, int j, int k, int dim) {
    return G_[(i * dim + j) * dim + k];
}

/* acceleration a^i = -Gamma^i_{jk} v^j v^k at position pos */
static void accel(const Manifold *m, const double *pos, const double *v,
                  double *a) {
    double *G_ = (double *)malloc(sizeof(double) * m->dim * m->dim * m->dim);
    m->christoffel(pos, G_, m->dim);
    for (int i = 0; i < m->dim; i++) {
        double s = 0.0;
        for (int j = 0; j < m->dim; j++)
            for (int k = 0; k < m->dim; k++)
                s += G(G_, i, j, k, m->dim) * v[j] * v[k];
        a[i] = -s;
    }
    free(G_);
}

void manifold_geodesic(const Manifold *m, const double *pos,
                       const double *vel, double T, int steps, double *result) {
    int n = m->dim;
    double dt = T / (double)steps;
    double *x = (double *)malloc(sizeof(double) * n);
    double *v = (double *)malloc(sizeof(double) * n);
    double *k1x = (double *)malloc(sizeof(double) * n);
    double *k1v = (double *)malloc(sizeof(double) * n);
    double *k2x = (double *)malloc(sizeof(double) * n);
    double *k2v = (double *)malloc(sizeof(double) * n);
    double *k3x = (double *)malloc(sizeof(double) * n);
    double *k3v = (double *)malloc(sizeof(double) * n);
    double *k4x = (double *)malloc(sizeof(double) * n);
    double *k4v = (double *)malloc(sizeof(double) * n);
    double *tx  = (double *)malloc(sizeof(double) * n);
    double *tv  = (double *)malloc(sizeof(double) * n);
    double *a   = (double *)malloc(sizeof(double) * n);

    memcpy(x, pos, sizeof(double) * n);
    memcpy(v, vel, sizeof(double) * n);

    for (int step = 1; step <= steps; step++) {
        /* k1 */
        memcpy(k1x, v, sizeof(double) * n);
        accel(m, x, v, k1v);

        /* k2 at midpoint */
        for (int i = 0; i < n; i++) { tx[i] = x[i] + 0.5 * dt * k1x[i];
                                      tv[i] = v[i] + 0.5 * dt * k1v[i]; }
        for (int i = 0; i < n; i++) k2x[i] = tv[i];
        accel(m, tx, tv, k2v);

        /* k3 at midpoint */
        for (int i = 0; i < n; i++) { tx[i] = x[i] + 0.5 * dt * k2x[i];
                                      tv[i] = v[i] + 0.5 * dt * k2v[i]; }
        for (int i = 0; i < n; i++) k3x[i] = tv[i];
        accel(m, tx, tv, k3v);

        /* k4 at endpoint */
        for (int i = 0; i < n; i++) { tx[i] = x[i] + dt * k3x[i];
                                      tv[i] = v[i] + dt * k3v[i]; }
        for (int i = 0; i < n; i++) k4x[i] = tv[i];
        accel(m, tx, tv, k4v);

        /* RK4 update */
        for (int i = 0; i < n; i++) {
            x[i] += dt / 6.0 * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]);
            v[i] += dt / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
        }
    }

    memcpy(result, x, sizeof(double) * n);

    free(x); free(v); free(k1x); free(k1v); free(k2x); free(k2v);
    free(k3x); free(k3v); free(k4x); free(k4v); free(tx); free(tv); free(a);
}

void manifold_exp(const Manifold *m, const double *pos,
                  const double *vel, double *result) {
    manifold_geodesic(m, pos, vel, 1.0, 200, result);
}

double manifold_geodesic_length(const Manifold *m, const double *pos,
                                const double *vel, double T, int steps) {
    int n = m->dim;
    double dt = T / (double)steps;
    double *x = (double *)malloc(sizeof(double) * n);
    double *v = (double *)malloc(sizeof(double) * n);
    double *k1x = (double *)malloc(sizeof(double) * n);
    double *k1v = (double *)malloc(sizeof(double) * n);
    double *k2x = (double *)malloc(sizeof(double) * n);
    double *k2v = (double *)malloc(sizeof(double) * n);
    double *k3x = (double *)malloc(sizeof(double) * n);
    double *k3v = (double *)malloc(sizeof(double) * n);
    double *k4x = (double *)malloc(sizeof(double) * n);
    double *k4v = (double *)malloc(sizeof(double) * n);
    double *tx  = (double *)malloc(sizeof(double) * n);
    double *tv  = (double *)malloc(sizeof(double) * n);
    double *a   = (double *)malloc(sizeof(double) * n);

    memcpy(x, pos, sizeof(double) * n);
    memcpy(v, vel, sizeof(double) * n);

    double length = 0.0;
    for (int step = 1; step <= steps; step++) {
        double vnorm = 0.0;
        for (int i = 0; i < n; i++) vnorm += v[i] * v[i];
        vnorm = sqrt(vnorm);
        length += vnorm * dt;

        memcpy(k1x, v, sizeof(double) * n);
        accel(m, x, v, k1v);
        for (int i = 0; i < n; i++) { tx[i] = x[i] + 0.5 * dt * k1x[i];
                                      tv[i] = v[i] + 0.5 * dt * k1v[i]; }
        for (int i = 0; i < n; i++) k2x[i] = tv[i];
        accel(m, tx, tv, k2v);
        for (int i = 0; i < n; i++) { tx[i] = x[i] + 0.5 * dt * k2x[i];
                                      tv[i] = v[i] + 0.5 * dt * k2v[i]; }
        for (int i = 0; i < n; i++) k3x[i] = tv[i];
        accel(m, tx, tv, k3v);
        for (int i = 0; i < n; i++) { tx[i] = x[i] + dt * k3x[i];
                                      tv[i] = v[i] + dt * k3v[i]; }
        for (int i = 0; i < n; i++) k4x[i] = tv[i];
        accel(m, tx, tv, k4v);
        for (int i = 0; i < n; i++) {
            x[i] += dt / 6.0 * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]);
            v[i] += dt / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
        }
    }

    free(x); free(v); free(k1x); free(k1v); free(k2x); free(k2v);
    free(k3x); free(k3v); free(k4x); free(k4v); free(tx); free(tv); free(a);
    return length;
}

/* --- S^2 in spherical coords (theta in [0,pi], phi in [0,2pi)) --- */
void sphere_christoffel(const double *pos, double *out, int dim) {
    (void)dim;
    double theta = pos[0];
    memset(out, 0, sizeof(double) * 2 * 2 * 2);
    double s = sin(theta), c = cos(theta);
    /* Gamma^theta_{phi,phi} = -sin theta cos theta */
    out[(0 * 2 + 1) * 2 + 1] = -s * c;
    /* Gamma^phi_{theta,phi} = Gamma^phi_{phi,theta} = cot theta */
    double cot = (s != 0.0) ? c / s : 0.0;
    out[(1 * 2 + 0) * 2 + 1] = cot;
    out[(1 * 2 + 1) * 2 + 0] = cot;
}

void make_sphere(Manifold *m, double radius) {
    (void)radius;
    m->dim = 2;
    m->christoffel = sphere_christoffel;
}

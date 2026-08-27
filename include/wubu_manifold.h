/*
 * wubu_manifold.h -- Generic Riemannian manifold geodesic integrator.
 * Ported math from tsotchke/quantum_geometric_tensor
 * src/quantum_geometric/distributed/differential_geometry.c (MIT): the RK4
 * integration of the geodesic equation
 *     dx^i/dt = v^i ,  dv^i/dt = -Gamma^i_{jk} v^j v^k
 * with Christoffel symbols Gamma. We strip qgt's distributed engine /
 * allocation framework and keep the reusable numerical core: any manifold
 * is described by a christoffel callback, and we integrate geodesics on it.
 *
 * Concrete validated manifold: the 2-sphere S^2 (great-circle geodesics),
 * used in src/tests/test_wubu_manifold.c to check exp/log/geodesic against
 * the closed-form arc length d = R * central_angle.
 */

#ifndef WUBU_MANIFOLD_H
#define WUBU_MANIFOLD_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int dim;
    /* Christoffel symbol Gamma[i][j][k] at coordinates pos[0..dim-1].
     * Must write into out[dim*dim*dim] in (i*dim + j)*dim + k order. */
    void (*christoffel)(const double *pos, double *out, int dim);
} Manifold;

/* Integrate the geodesic starting at `pos` with initial velocity `vel`
 * (tangent vector) for total parameter `T`, using `steps` RK4 steps.
 * Fills `result` (length dim) with the endpoint and returns its length.
 * `result` must point to dim doubles. */
void manifold_geodesic(const Manifold *m, const double *pos,
                       const double *vel, double T, int steps, double *result);

/* Riemannian exponential map: exp_p(v) = endpoint of geodesic from p with
 * initial velocity v, integrated for unit parameter. */
void manifold_exp(const Manifold *m, const double *pos,
                  const double *vel, double *result);

/* Arc length of the integrated geodesic = integral |v| dt (RK4 accumulates
 * segment lengths). Returns the total length. */
double manifold_geodesic_length(const Manifold *m, const double *pos,
                                const double *vel, double T, int steps);

/* --- Concrete manifolds (for tests / reuse) --- */

/* 2-sphere S^2 of radius R in spherical coords (theta, phi).
 * Christoffel (nonzero): Gamma^theta_{phi,phi} = -sin theta cos theta
 *                         Gamma^phi_{theta,phi} = Gamma^phi_{phi,theta} = cot theta */
void sphere_christoffel(const double *pos, double *out, int dim);
void make_sphere(Manifold *m, double radius);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MANIFOLD_H */

/*
 * wubu_so3.h -- SO(3) Lie-group exp / log / geodesic, ported from
 * tsotchke/libirrep (MIT) and adapted to WuBuMath's Quat [w,x,y,z] / float
 * convention.
 *
 * Algorithms (faithful port of libirrep/src/so3.c):
 *   - rot_exp: Rodrigues formula with small-angle guard (IRREP_ROT_EXP_SMALL)
 *   - rot_log: quaternion-routed (Shepperd + axis-angle) for pi-safety
 *   - geodesic_distance: angle of R_a^{-1} R_b  (Riemannian distance on SO(3))
 *   - rot_compose / rot_inverse: matrix group ops (inverse = transpose)
 *
 * Cross-checked numerically in src/tests/test_wubu_so3.c:
 *   rot_exp(rot_log(R)) == R  to 1e-5, and geodesic_distance(rot_exp(w), I)
 *   == |w| to 1e-5, for random w and random rotations.
 *
 * Why this exists: WuBuMath/lean/WubuProofs/FiberBundle.lean previously had a
 * fake `bundle_projection` returning 0 and no real exp/log on the rotation
 * group. This C module supplies the genuine manifold maps; the Lean side can
 * later state theorems against these numerics.
 */

#ifndef WUBU_SO3_H
#define WUBU_SO3_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 3x3 rotation matrix, row-major (m[0..2] = row 0). */
typedef struct {
    float m[9];
} Rot3;

/* Axis-angle: unit axis[3], angle in radians. */
typedef struct {
    float axis[3];
    float angle;
} AxisAngle;

/* --- Group ops --- */
Rot3 rot_identity(void);
Rot3 rot_compose(Rot3 A, Rot3 B);
Rot3 rot_inverse(Rot3 R);

/* --- Lie exp / log --- */
Rot3      rot_exp(const float omega[3]);          /* so(3) -> SO(3) */
void      rot_log(Rot3 R, float omega_out[3]);    /* SO(3) -> so(3) */

/* --- Geodesic --- */
float     rot_geodesic_distance(Rot3 A, Rot3 B);  /* Riemannian angle */

/* --- Round-trip helpers used by tests --- */
AxisAngle rot_to_axis_angle(Rot3 R);
Rot3      rot_from_axis_angle(const float axis[3], float angle);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SO3_H */

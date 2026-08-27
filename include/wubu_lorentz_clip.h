/* GAP-C042: Lorentz-model CLIP similarity primitives */
#ifndef WUBU_LORENTZ_CLIP_H
#define WUBU_LORENTZ_CLIP_H
#ifdef __cplusplus
extern "C" {
#endif
/* Minkowski inner product (time-first convention). */
float wubu_lc_minkowski_ip(const float* a,const float* b,int D);
/* arccosh(-<a,b>_L) — Lorentzian distance on the hyperboloid. */
float wubu_lc_distance(const float* a,const float* b,int D);
/* lift Euclidean vector to hyperboloid; out has D=Ds+1 components. */
void wubu_lc_lift(const float* v,int Ds,float c,float* out);
/* fix time component so L(p,p)=-1/c. */
void wubu_lc_project(float* p,int Ds,float c);
#ifdef __cplusplus
}
#endif
#endif

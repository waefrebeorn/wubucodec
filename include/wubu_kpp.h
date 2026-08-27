/* GAP-D033: hyperbolic k-means++ seeding */
#ifndef WUBU_KPP_H
#define WUBU_KPP_H
#ifdef __cplusplus
extern "C" {
#endif
/* D^2 seeding over geodesic distances; out_centers: [k] point indices. */
int wubu_kpp_seed(const float* pts,int n,int D,int k,float c,
                   unsigned* seed,int* out_centers);
#ifdef __cplusplus
}
#endif
#endif

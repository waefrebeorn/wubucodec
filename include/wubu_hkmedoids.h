/* GAP-D025: hyperbolic k-medoids (PAM) */
#ifndef WUBU_HKMEDOIDS_H
#define WUBU_HKMEDOIDS_H
#ifdef __cplusplus
extern "C" {
#endif
/* medoids: [k] point indices; assign: [n] medoid index per point. */
int wubu_hkmedoids(const float* pts,int n,int D,int k,float c,
                   int iters,int* medoids,int* assign);
#ifdef __cplusplus
}
#endif
#endif

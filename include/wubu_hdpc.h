/* GAP-D021: hyperbolic density-peak clustering */
#ifndef WUBU_HDPC_H
#define WUBU_HDPC_H
#ifdef __cplusplus
extern "C" {
#endif
/* rho: [n] densities; assign: [n] center index per point;
 * is_center: [n] flags. dc_frac = cutoff percentile (e.g. 0.02). */
int wubu_hdpc_run(const float* pts,int n,int D,float c,float dc_frac,
                  int* rho,int* assign,int* is_center);
#ifdef __cplusplus
}
#endif
#endif

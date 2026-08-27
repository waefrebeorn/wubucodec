/* GAP-D019: hyperbolic MDS */
#ifndef WUBU_HMDS_H
#define WUBU_HMDS_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_hmds_stress(const float* pts,int n,int D,float c,
                        const float* target);
int  wubu_hmds_embed(const float* target,int n,int D,float c,
                     int iters,float lr,unsigned seed,float* out);
#ifdef __cplusplus
}
#endif
#endif

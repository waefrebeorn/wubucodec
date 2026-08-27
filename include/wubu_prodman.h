/* GAP-D014: product manifold (hyperbolic × euclidean) */
#ifndef WUBU_PRODMAN_H
#define WUBU_PRODMAN_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_pm_hyper_dist(const float* a,const float* b,int D,float c);
/* x,y: [D_hyp+D_euc]. Distance = sqrt(d_hyp² + d_euc²). */
float wubu_pm_product_dist(const float* x,const float* y,
                            int D_hyp,int D_euc,float c);
void wubu_pm_project(float* pt,int D_hyp,int D_euc,float c);
void wubu_pm_init_random(float* pts,int n,int D_hyp,int D_euc,
                          float c,unsigned* seed);
#ifdef __cplusplus
}
#endif
#endif

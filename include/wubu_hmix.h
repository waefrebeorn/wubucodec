/* GAP-D029: hyperbolic-spherical product manifold */
#ifndef WUBU_HMIX_H
#define WUBU_HMIX_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_hm_hyper_dist(const float* a,const float* b,int Dh,float c);
/* x,y: [Dh+Ds] with hyperbolic coords [0,Dh) and spherical [Dh,Dh+Ds). */
float wubu_hm_product_dist(const float* x,const float* y,
                            int Dh,int Ds,float c);
void wubu_hm_project(float* pt,int Dh,int Ds,float c);
#ifdef __cplusplus
}
#endif
#endif

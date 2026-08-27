/* GAP-C026: Euclidean parametrization of hyperbolic space */
#ifndef WUBU_EUCLIDPARAM_H
#define WUBU_EUCLIDPARAM_H
#ifdef __cplusplus
extern "C" {
#endif
void  wubu_ep_to_ball(const float* z,int D,float c,float* x);
void  wubu_ep_from_ball(const float* x,int D,float c,float* z);
void  wubu_ep_sgd_step(float* z,const float* grad,int D,float lr);
float wubu_ep_distance(const float* z1,const float* z2,int D,float c);
#ifdef __cplusplus
}
#endif
#endif

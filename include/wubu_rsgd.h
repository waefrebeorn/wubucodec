/* GAP-C023: Riemannian SGD on the Poincaré ball */
#ifndef WUBU_RSGD_H
#define WUBU_RSGD_H
#ifdef __cplusplus
extern "C" {
#endif
/* x: [D] current position (modified in-place), grad: [D] gradient.
 * Returns new norm. */
float wubu_rsgd_step(float* x,const float* grad,int D,float c,float lr);
void wubu_rsgd_batch(float* xs,int N,int D,float c,float lr,
                     const float* grads);
#ifdef __cplusplus
}
#endif
#endif

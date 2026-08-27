/* GAP-C024: Hyperbolic VAE wrapped-normal sampling */
#ifndef WUBU_HVAE_H
#define WUBU_HVAE_H
#ifdef __cplusplus
extern "C" {
#endif
#include <math.h>
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

/* sample z ~ WrappedNormal(mu, sigma) on the Poincaré ball */
void wubu_hvae_sample(float* z,const float* mu,const float* sigma,
                      int D,float c);
/* Jacobian-corrected log-density for VAE KL computation. */
float wubu_hvae_log_density(const float* z,const float* mu,
                             const float* sigma,int D,float c);
#ifdef __cplusplus
}
#endif
#endif

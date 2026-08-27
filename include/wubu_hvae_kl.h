/* GAP-D028: hyperbolic VAE KL divergence (Monte-Carlo) */
#ifndef WUBU_HVAE_KL_H
#define WUBU_HVAE_KL_H
#ifdef __cplusplus
extern "C" {
#endif
/* KL( q(z)=WN(mu_q,sigma_q) || p(z)=WN(mu_p,sigma_p) ) via MC sampling. */
float wubu_hvae_kl_estimate(const float* mu_q,const float* sigma_q,
                             const float* mu_p,const float* sigma_p,
                             int D,float c,int n_samples);
#ifdef __cplusplus
}
#endif
#endif

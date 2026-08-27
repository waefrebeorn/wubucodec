/* GAP-H015: hyperbolic PPO clipped surrogate */
#ifndef WUBU_HPPO_H
#define WUBU_HPPO_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_hp_ratio(const float* mu_new,const float* mu_old,
                     const float* state,int D,float c,float sigma);
float wubu_hp_surrogate(const float* mu_new,const float* mu_old,
                         const float* states,const float* advantages,
                         int N,int D,float c,float sigma,float eps);
#ifdef __cplusplus
}
#endif
#endif

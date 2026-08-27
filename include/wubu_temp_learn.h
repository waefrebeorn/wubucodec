/* GAP-D007: learned temperature contrastive */
#ifndef WUBU_TEMP_LEARN_H
#define WUBU_TEMP_LEARN_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_tl_logit(const float* a,const float* b,int D,float c,float log_tau);
float wubu_tl_infonce(const float* anchor,const float* positive,
                       const float* negatives,int n_neg,
                       int D,float c,float* log_tau,float lr);
#ifdef __cplusplus
}
#endif
#endif

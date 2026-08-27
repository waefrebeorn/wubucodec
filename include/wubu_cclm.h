/* GROUP 10: CCLM + JCCR */
#ifndef WUBU_CCLM_H
#define WUBU_CCLM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int    wubu_cclm_fit(const uint8_t* luma,const uint8_t* chroma,
                       int n_samples,double* alpha,double* beta);
void   wubu_cclm_predict(const uint8_t* luma_block,int bs,
                          double alpha,double beta,uint8_t* chroma_out);
int    wubu_cclm_best_mode(const uint8_t* luma_ref,const uint8_t* chroma_ref,
                             const uint8_t* actual_chroma,int bs,
                             double* out_alpha,double* out_beta);
int    wubu_jccr_check(const int16_t* res_cb,const int16_t* res_cr,
                          int n_coeffs,float threshold_ratio);
void   wubu_jccr_compute(const int16_t* res_cb,const int16_t* res_cr,
                           int16_t* res_joint,int n_coeffs);
void   wubu_jccr_split(const int16_t* res_joint,int16_t* res_cb,int16_t* res_cr,
                         int n_coeffs,int mode);
#ifdef __cplusplus
}
#endif
#endif

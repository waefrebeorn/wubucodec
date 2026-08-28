/* GROUP 8: Trellis quantization */
#ifndef WUBU_TRELLIS_H
#define WUBU_TRELLIS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
double wubu_trellis_quantize(const double* coeffs,int n_coeffs,
                              double qstep,double lambda,int16_t* output_levels);
double wubu_trellis_quantize_fast(const double* coeffs,int n_coeffs,
                              double qstep,double lambda,int16_t* output_levels);
long   wubu_trellis_vs_rounding(const double* coeffs,int n_coeffs,
                                  double qstep,double lambda,
                                  long* out_sse_std,long* out_bits_std,
                                  long* out_sse_tr,long* out_bits_tr);
int    tr_bits_for_level(int level);
#ifdef __cplusplus
}
#endif
#endif

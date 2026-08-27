/* GROUP 7: Variable-size transforms (C9) */
#ifndef WUBU_TRANSFORM_H
#define WUBU_TRANSFORM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Existing API (backward compatible) */
void wubu_tr_forward(const int16_t* input,int16_t* output,int size);
void wubu_tr_inverse(const int16_t* coeff,int16_t* output,int size);
void wubu_tr_skip(const int16_t* input,int16_t* output,int size);
void wubu_tr_quantize_m(const int16_t* coeffs,const uint8_t* qmat,
                          int16_t* output,int size,int qp);
void wubu_tr_dequantize_m(const int16_t* quantized,const uint8_t* qmat,
                            int16_t* output,int size,int qp);
const uint8_t* wubu_tr_get_qmat(int size,int intra);

/* C9: multi-size transform API */
/* Available transform sizes: 4, 8, 16, 32 (powers of 2) */
int wubu_tr_get_sizes(int* sizes,int max);
int wubu_tr_forward_size(int size,const int16_t* input,int16_t* output);
int wubu_tr_inverse_size(int size,const int16_t* coeff,int16_t* output);
int wubu_tr_forward_dst7_4x4(const int16_t* input,int16_t* output);  /* 4x4 DST-VII forward */
int wubu_tr_inverse_dst7_4x4(const int16_t* coeff,int16_t* output); /* 4x4 DST-VII inverse */

/* C9: quantization matrices per size (JPEG-style for visual, flat for others) */
const uint8_t* wubu_tr_get_qmat_size(int size,int intra);
int wubu_tr_size_to_index(int size); /* map size to quantization matrix index */

#ifdef __cplusplus
}
#endif
#endif

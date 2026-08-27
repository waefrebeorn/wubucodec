/* GAP-C070: integer 8x8 DCT + quantization */
#ifndef WUBU_IDCT8X8_H
#define WUBU_IDCT8X8_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_dct8x8_forward(const int* input,int* output);
void wubu_dct8x8_inverse(const int* input,int* output);
void wubu_dct8x8_quantize(const int* coeffs,int quality,int* quantized);
void wubu_dct8x8_dequantize(const int* quantized,int quality,int* coeffs);
#ifdef __cplusplus
}
#endif
#endif

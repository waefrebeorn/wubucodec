/* GAP-H016: Bellman operator (contraction + fixed point) */
#ifndef WUBU_BELLMAN_H
#define WUBU_BELLMAN_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void wubu_bellman_apply(float* V,const int8_t* next,const float* R,
                         int S,int A,float gamma,float* out);
float wubu_bellman_supdiff(const float* a,const float* b,int S);
int  wubu_bellman_contraction_check(const int8_t* next,const float* R,
                                     int S,int A,float gamma,
                                     unsigned seed);
#ifdef __cplusplus
}
#endif
#endif

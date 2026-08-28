/* wubu_satd.h — SATD for mode decision */
#ifndef WUBU_SATD_H
#define WUBU_SATD_H
#include <stdint.h>

int wubu_satd_4x4(const uint8_t* a, const uint8_t* b, int stride_a, int stride_b);
int wubu_satd_8x8(const uint8_t* a, const uint8_t* b, int stride_a, int stride_b);
int wubu_satd_fast_8x8(const uint8_t* a, const uint8_t* b, int stride_a, int stride_b);

#endif

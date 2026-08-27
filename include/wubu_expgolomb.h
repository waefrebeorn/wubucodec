/* GAP-E008: exponential-Golomb bit-level coding */
#ifndef WUBU_EXPGOLOMB_H
#define WUBU_EXPGOLOMB_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {uint8_t* buf;size_t cap,pos;int bit;} WubuBitWriter;
typedef struct {const uint8_t* buf;size_t len,pos;int bit;} WubuBitReader;

void   wubu_bw_init(WubuBitWriter* bw,uint8_t* buf,size_t cap);
int    wubu_eg_put(WubuBitWriter* bw,uint32_t v);
void   wubu_br_init(WubuBitReader* br,const uint8_t* buf,size_t len);
int    wubu_eg_get(WubuBitReader* br,uint32_t* out);
#ifdef __cplusplus
}
#endif
#endif

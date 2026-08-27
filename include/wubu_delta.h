/* GAP-E007: delta + zigzag + varint compression */
#ifndef WUBU_DELTA_H
#define WUBU_DELTA_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t wubu_zigzag_encode(int32_t v);
int32_t  wubu_zigzag_decode(uint32_t v);
void     wubu_delta_encode(int32_t* vals,int n);
void     wubu_delta_decode(int32_t* vals,int n);
size_t   wubu_varint_put(uint8_t* buf,uint32_t v);
size_t   wubu_varint_get(const uint8_t* buf,size_t max,uint32_t* out);
size_t   wubu_delta_compress(const int32_t* src,int n,uint8_t* buf,size_t cap);
int      wubu_delta_decompress(const uint8_t* buf,size_t len,
                                int32_t* dst,int n);
#ifdef __cplusplus
}
#endif
#endif

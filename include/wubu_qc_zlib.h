/* GAP-C059: zlib entropy stage for quaternion codec */
#ifndef WUBU_QC_ZLIB_H
#define WUBU_QC_ZLIB_H
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
long wubu_qz_compress(const uint8_t* src,long src_size,uint8_t** out_buf);
long wubu_qz_decompress(const uint8_t* src,long src_size,
                          uint8_t* dest,long dest_size);
long wubu_qz_encode(const uint8_t* frames,int n_frames,int W,int H,
                     float angle_step,FILE* out);
void wubu_qz_decode(FILE* in,uint8_t* frames_out,
                     int n_frames,int W,int H,float angle_step);
#ifdef __cplusplus
}
#endif
#endif

/* GAP-C056: complete quaternion video codec */
#ifndef WUBU_QUAT_CODEC_H
#define WUBU_QUAT_CODEC_H
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int W,H;
    float target_bpf;
    int frame_count;
    uint8_t* prev_frame;
    int has_prev;
    long total_bytes;
    unsigned seed;
} WubuQC;

int  wubu_qc_init(WubuQC* qc,int W,int H,float target_bpf,unsigned seed);
void wubu_qc_free(WubuQC* qc);
long wubu_qc_encode_frame(WubuQC* qc,const uint8_t* frame,FILE* out);
int  wubu_qc_decode_frame(WubuQC* qc,FILE* in,uint8_t* frame_out);
#ifdef __cplusplus
}
#endif
#endif

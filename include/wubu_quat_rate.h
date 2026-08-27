/* GAP-C050: adaptive quaternion rate control */
#ifndef WUBU_QUAT_RATE_H
#define WUBU_QUAT_RATE_H
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
float wubu_qr_angular_velocity(const float* q_prev,const float* q_curr);
void  wubu_qr_allocate(const float* angles,int n_frames,
                        int target_bytes_per_frame,
                        int min_bits,int max_bits,int* out_bits);
int   wubu_qr_encode_delta(float angle,float ax,float ay,float az,
                            int n_bits,FILE* f);
float wubu_qr_decode_delta(FILE* f,int n_bits,
                            float* axis_x,float* axis_y,float* axis_z);
long  wubu_qr_encode_sequence(const float* quat_frames,int n_frames,int D,
                               float target_bpf,const char* path);
#ifdef __cplusplus
}
#endif
#endif

/* GROUP 1: DPB + Weighted prediction + Bi-prediction */
#include <stddef.h>
#ifndef WUBU_DPB_H
#define WUBU_DPB_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    uint8_t* frames;    /* max_frames × W×H×3 */
    int* poc;           /* picture order count per frame */
    int max_frames;
    int count;
    int head;
    size_t frame_size;
} DPB;

DPB* wubu_dpb_create(int max_frames,int W,int H);
void wubu_dpb_destroy(DPB* dpb);
int  wubu_dpb_push(DPB* dpb,const uint8_t* frame,int poc,int W,int H);
const uint8_t* wubu_dpb_get(DPB* dpb,int ref_idx);
int  wubu_dpb_get_poc(DPB* dpb,int ref_idx);
int  wubu_dpb_count(DPB* dpb);
long wubu_dpb_multi_me(const uint8_t* curr,const uint8_t* predicted,
                         DPB* dpb,int W,int H,int bx,int by,int bs,
                         int search_range,
                         int* out_ref_idx,int* out_dx,int* out_dy);
void wubu_weighted_pred(const uint8_t* ref0,const uint8_t* ref1,
                          uint8_t* output,long n_pixels,
                          int w0,int w1,int offset,int shift);
void wubu_bipred_average(const uint8_t* p0,const uint8_t* p1,
                           uint8_t* output,long n_pixels);
#ifdef __cplusplus
}
#endif
#endif

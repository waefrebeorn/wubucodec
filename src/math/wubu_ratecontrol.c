/* wubu_ratecontrol.c — CRF rate control with per-frame QP adjustment */
#include "wubu_ratecontrol.h"
#include <math.h>
#include <string.h>

/* CRF: constant rate factor — adjust QP per frame to maintain quality */
/* Higher CRF = lower quality, lower bitrate */

void wubu_rc_init(RateControl* rc, int target_bitrate, float crf){
    rc->target_bitrate = target_bitrate;
    rc->crf = crf;
    rc->frame_count = 0;
    rc->total_bits = 0;
    rc->avg_qp = 26;
    rc->buffer_fullness = 0;
}

/* Get QP for next frame based on CRF and content complexity */
int wubu_rc_get_qp(RateControl* rc, int frame_type, float complexity){
    /* Base QP from CRF */
    int base_qp = (int)(rc->crf + 0.5f);
    
    /* Adjust for content complexity (higher complexity → lower QP) */
    float complexity_factor = 1.0f;
    if(complexity > 100.0f) complexity_factor = 0.9f;
    if(complexity > 500.0f) complexity_factor = 0.8f;
    if(complexity < 50.0f) complexity_factor = 1.1f;
    
    int qp = (int)(base_qp * complexity_factor);
    
    /* Frame type offsets */
    if(frame_type == WUBU_P_FRAME) qp += 2;
    if(frame_type == WUBU_B_FRAME) qp += 4;
    
    /* Clamp */
    if(qp < 0) qp = 0;
    if(qp > 51) qp = 51;
    
    return qp;
}

/* Update rate control after encoding a frame */
void wubu_rc_update(RateControl* rc, long frame_bits, int frame_qp){
    rc->frame_count++;
    rc->total_bits += frame_bits;
    /* Running average QP */
    rc->avg_qp = (rc->avg_qp * 9 + frame_qp) / 10;
}

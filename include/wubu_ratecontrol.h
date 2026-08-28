/* wubu_ratecontrol.h — CRF rate control */
#ifndef WUBU_RATECONTROL_H
#define WUBU_RATECONTROL_H
#include <stdint.h>
#include "wubu_rdo.h"

typedef struct {
    int target_bitrate;   /* target bitrate (0 = CRF mode) */
    float crf;            /* constant rate factor (0-51) */
    int frame_count;
    long total_bits;
    int avg_qp;
    int buffer_fullness;
} RateControl;

void wubu_rc_init(RateControl* rc, int target_bitrate, float crf);
int  wubu_rc_get_qp(RateControl* rc, int frame_type, float complexity);
void wubu_rc_update(RateControl* rc, long frame_bits, int frame_qp);

#endif

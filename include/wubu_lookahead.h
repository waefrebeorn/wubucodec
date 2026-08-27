/* Lookahead buffer + VBV rate control */
#ifndef WUBU_LOOKAHEAD_H
#define WUBU_LOOKAHEAD_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    double buffer_size,current_fill,bitrate,framerate;
} WubuVbv;
void wubu_vbv_init(WubuVbv* v,double buf,double br,double fps);
int  wubu_vbv_can_decode(const WubuVbv* v,long bits);
void wubu_vbv_update(WubuVbv* v,long actual_bits);
long wubu_vbv_max_frame_bits(const WubuVbv* v);

typedef struct {
    uint8_t** frames;double* complexities;int* is_scene_change;
    int capacity,count,head;
} Lookahead;
Lookahead* wubu_la_create(int depth,int W,int H);
void wubu_la_destroy(Lookahead* la);
void wubu_la_push(Lookahead* la,const uint8_t* frame,int W,int H);
double wubu_la_avg_complexity(const Lookahead* la);
int  wubu_la_has_scenecut(const Lookahead* la);
int  wubu_la_adjust_qp(const Lookahead* la,int base_qp,double crf_factor);

/* re-export from scene.c */
extern int wubu_scene_change(const uint8_t*,const uint8_t*,int,int,double);
extern double wubu_temporal_complexity(const uint8_t*,const uint8_t*,long);
extern double wubu_spatial_complexity(const uint8_t*,int,int);
#ifdef __cplusplus
}
#endif
#endif

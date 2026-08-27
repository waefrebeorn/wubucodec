/* GROUP 16: Scene analysis & adaptive encoding */
#ifndef WUBU_SCENE_H
#define WUBU_SCENE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void   wubu_histogram(const uint8_t* img,long n,uint32_t* hist);
double wubu_hist_distance(const uint32_t* h1,const uint32_t* h2,long total);
int    wubu_scene_change(const uint8_t* prev,const uint8_t* curr,
                          int W,int H,double threshold);
double wubu_spatial_complexity(const uint8_t* img,int W,int H);
double wubu_temporal_complexity(const uint8_t* prev,const uint8_t* curr,long n);
int    wubu_adaptive_qp(double spatial_act,double temporal_act,
                          int base_qp,int max_offset);

typedef struct {
    double target_bits_per_pixel;
    int last_qp;
    double quality_factor;
} WubuCrfState;
void wubu_crf_init(WubuCrfState* crf,double crf_value);
int  wubu_crf_update(WubuCrfState* crf,long actual_bits,
                      long target_bits,int current_qp);
#ifdef __cplusplus
}
#endif
#endif

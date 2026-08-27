/* GROUP 15: Hierarchical B-frame structure */
#ifndef WUBU_BFRAME2_H
#define WUBU_BFRAME2_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {WUBU_FT_I,WUBU_FT_P,WUBU_FT_B} wubu_frame_type2_t;

int wubu_gop_temporal_layer(int display_index,int gop_size);
wubu_frame_type2_t wubu_gop_frame_type(int display_index,int gop_size);
int wubu_gop_coding_order(int display_index,int gop_size);
void wubu_bp_average(const uint8_t* p0,const uint8_t* p1,
                      uint8_t* output,long n_pixels);
void wubu_bp_weighted(const uint8_t* p0,const uint8_t* p1,
                        uint8_t* output,long n_pixels,
                        int w0,int w1,int weight_denom);

#ifdef __cplusplus
}
#endif
#endif

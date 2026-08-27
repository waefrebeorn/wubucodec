/* GROUP 22: Screen content coding */
#ifndef WUBU_SCREENC_H
#define WUBU_SCREENC_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
long wubu_ibc_search(const uint8_t* recon,int W,int H,
                      int bx,int by,int bs,int search_range,
                      int* out_bx,int* out_by);
void wubu_ibc_predict(const uint8_t* recon,int W,int H,
                       int src_bx,int src_by,int dst_bx,int dst_by,int bs,
                       uint8_t* output);
int  wubu_palette_extract(const uint8_t* img,int W,int H,
                            int bx,int by,int bs,
                            uint8_t* palette,int max_colors);
int  wubu_palette_encode(const uint8_t* img,int W,int H,
                           int bx,int by,int bs,
                           const uint8_t* palette,int n_colors,
                           uint8_t* indices);
void wubu_palette_decode(const uint8_t* indices,const uint8_t* palette,
                           int n_colors,uint8_t* output,int n_pixels);
int  wubu_palette_suitable(const uint8_t* img,int W,int H,
                             int bx,int by,int bs,int max_colors);
#ifdef __cplusplus
}
#endif
#endif

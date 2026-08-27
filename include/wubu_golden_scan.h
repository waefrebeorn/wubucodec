/* GAP-A012+A010: golden-angle progressive scan + locality metric */
#ifndef WUBU_GOLDEN_SCAN_H
#define WUBU_GOLDEN_SCAN_H
#ifdef __cplusplus
extern "C" {
#endif
void  wubu_gs_order(int n,float* out_x,float* out_y);
float wubu_gs_locality(const float* xs,const float* ys,int n);
void  wubu_gs_raster(int n_side,float* out_x,float* out_y);
#ifdef __cplusplus
}
#endif
#endif

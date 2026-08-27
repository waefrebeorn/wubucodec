/* GAP-A011: Hilbert-curve scan order (locality baseline) */
#ifndef WUBU_HILBERT_H
#define WUBU_HILBERT_H
#ifdef __cplusplus
extern "C" {
#endif
void  wubu_hil_order(int side,float* out_x,float* out_y);
float wubu_hil_locality(const float* xs,const float* ys,int n);
#ifdef __cplusplus
}
#endif
#endif

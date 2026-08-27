/* GAP-C066: optimal K-segmentation via DP */
#ifndef WUBU_KSEG_H
#define WUBU_KSEG_H
#ifdef __cplusplus
extern "C" {
#endif
int wubu_seg_optimal(const float* quats,int n_frames,int D,int k,
                      int* out_indices);
#ifdef __cplusplus
}
#endif
#endif

/* GAP-C065: RD-optimized keyframe selection */
#ifndef WUBU_KEYSELECT_H
#define WUBU_KEYSELECT_H
#ifdef __cplusplus
extern "C" {
#endif
int   wubu_kf_select(const float* quats,int n_frames,int D,int n_keys,
                      int* out_indices);
float wubu_kf_estimate_error(const float* quats,int n_frames,int D,
                              const int* key_indices,int n_keys);
#ifdef __cplusplus
}
#endif
#endif

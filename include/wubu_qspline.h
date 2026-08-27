/* GAP-C064: quaternion Catmull-Rom spline trajectory */
#ifndef WUBU_QSPLINE_H
#define WUBU_QSPLINE_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_qsp_slerp(const float* qa,const float* qb,float t,float* out);
void wubu_qsp_catmull(const float* q0,const float* q1,
                       const float* q2,const float* q3,
                       float t,float* out);
void wubu_qsp_trajectory(const float* keys,int n_keys,
                          float t_global,float* out);
#ifdef __cplusplus
}
#endif
#endif

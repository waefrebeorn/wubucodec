/* GAP-C062: dual quaternion motion model */
#ifndef WUBU_DUALQUAT_H
#define WUBU_DUALQUAT_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_dq_create(float angle,float ax,float ay,float az,
                     float tx,float ty,float tz,float* out);
void wubu_dq_get_translation(const float* dq,float* t);
void wubu_dq_transform(const float* dq,const float* point,float* result);
void wubu_dq_slerp(const float* qa,const float* qb,float t,float* out);
#ifdef __cplusplus
}
#endif
#endif

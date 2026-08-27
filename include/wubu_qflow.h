/* GAP-C001: quaternion Hamilton-product-native flow field */
#ifndef WUBU_QFLOW_H
#define WUBU_QFLOW_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_qf_velocity(const float* q_current,const float* q_target,
                       float dt,float* velocity);
int  wubu_qf_step(const float* q_current,const float* velocity,
                   float dt,float* q_next);
int  wubu_qf_trajectory(const float* q_start,const float* q_end,
                         int n_steps,float* path);
#ifdef __cplusplus
}
#endif
#endif

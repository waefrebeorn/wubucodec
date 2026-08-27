/* GAP-C028: quaternion SLERP path for VHF cell motion */
#ifndef WUBU_SLERP_PATH_H
#define WUBU_SLERP_PATH_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_q_normalize(float* q);
void wubu_q_slerp(float* out,const float* qa,const float* qb,float t);
void wubu_q_path(const float* q0,const float* q1,int steps,float* out);
void wubu_cell_path(float* out,const float* cell0,const float* cell1,
                    int steps);
#ifdef __cplusplus
}
#endif
#endif

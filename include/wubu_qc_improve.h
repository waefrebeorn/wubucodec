/* GAP-C057: constant-velocity SLERP prediction */
#ifndef WUBU_QC_IMPROVE_H
#define WUBU_QC_IMPROVE_H
#ifdef __cplusplus
extern "C" {
#endif
void  wubu_qi_predict(const float* q_prev,const float* q_curr,
                       float* q_predicted);
float wubu_qi_angle(const float* a,const float* b);
float wubu_qi_prediction_error(const float* q_predicted,const float* q_actual);
#ifdef __cplusplus
}
#endif
#endif

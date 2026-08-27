/* GAP-C049: learning-rate schedules with conformal damping */
#ifndef WUBU_LRSCHED_H
#define WUBU_LRSCHED_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_lr_warmup_cosine(int t,int T_warmup,int T_total,
                             float eta_max,float eta_min);
float wubu_lr_damped(float base_lr,const float* x,int D,float c,
                      float max_damp);
float wubu_lr_schedule_step(int t,int T_warmup,int T_total,
                             float eta_max,float eta_min,
                             const float* x,int D,float c,
                             float max_damp);
#ifdef __cplusplus
}
#endif
#endif

/* GAP-H014: hyperbolic TD learning */
#ifndef WUBU_HTD_H
#define WUBU_HTD_H
#ifdef __cplusplus
extern "C" {
#endif
/* one TD step toward target with conformal damping */
void wubu_htd_step(float* value,const float* target,int D,float c,
                   float lr,float gamma);
/* iterate to convergence; returns steps, or -steps if diverged. */
int  wubu_htd_converge(float* value,const float* target,int D,float c,
                       float lr,float gamma,float eps,int max_steps);
#ifdef __cplusplus
}
#endif
#endif

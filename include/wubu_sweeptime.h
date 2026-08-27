/* GAP-A006: sweep time-axis model */
#ifndef WUBU_SWEEPTIME_H
#define WUBU_SWEEPTIME_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_st_x_to_t(int x,int width,float duration);
int   wubu_st_t_to_x(float t,float duration,int width);
int   wubu_st_t_to_frame(float t,float fps);
#ifdef __cplusplus
}
#endif
#endif

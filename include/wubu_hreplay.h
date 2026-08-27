/* GAP-H013: hyperbolic prioritized experience replay */
#ifndef WUBU_HREPLAY_H
#define WUBU_HREPLAY_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int cap,len,head,D;
    float c,div_w;
    float* states;  /* [cap, D] */
    float* prio;    /* [cap] */
} WubuRP;

int  wubu_rp_init(WubuRP* r,int cap,int D,float c,float div_weight);
void wubu_rp_free(WubuRP* r);
void wubu_rp_add(WubuRP* r,const float* state,float td_error);
int  wubu_rp_sample(const WubuRP* r,unsigned* seed);
void wubu_rp_update_prio(WubuRP* r,int idx,float td_error);
#ifdef __cplusplus
}
#endif
#endif

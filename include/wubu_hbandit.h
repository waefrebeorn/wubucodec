/* GAP-H012: hyperbolic contextual bandit */
#ifndef WUBU_HBANDIT_H
#define WUBU_HBANDIT_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int n,D;float tau;
    float* proto;   /* [n, D] action prototypes on ball */
    float* alpha;   /* [n] Beta positive counts */
    float* beta;    /* [n] Beta negative counts */
    unsigned seed;
} WubuHB;

int  wubu_hb_init(WubuHB* b,int n_actions,int D,float tau,unsigned seed);
void wubu_hb_free(WubuHB* b);
int  wubu_hb_select(WubuHB* b,const float* context,float c);
void wubu_hb_update(WubuHB* b,int action,const float* context,
                    float reward,float c);
float wubu_hb_mean_reward(const WubuHB* b,int action);
#ifdef __cplusplus
}
#endif
#endif

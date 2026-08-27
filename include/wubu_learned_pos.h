/* GAP-C034: learned positional embedding table (hyperbolic add) */
#ifndef WUBU_LEARNED_POS_H
#define WUBU_LEARNED_POS_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int max_T,D;
    float* table; /* [max_T,D] trainable */
} WubuLearnedPos;

int  wubu_lp_init(WubuLearnedPos* lp,int max_T,int D,unsigned seed);
void wubu_lp_free(WubuLearnedPos* lp);
const float* wubu_lp_row(const WubuLearnedPos* lp,int t);
void wubu_lp_apply(const WubuLearnedPos* lp,const float* tok,
                   int t,float c,float* out);
void wubu_lp_train_row(WubuLearnedPos* lp,int t,const float* grad,float lr);
#ifdef __cplusplus
}
#endif
#endif

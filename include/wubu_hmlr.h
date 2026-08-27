/* GAP-C019: prototype-based hyperbolic MLR (Poincare ball) */
#ifndef WUBU_HMLR_H
#define WUBU_HMLR_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int K,D;float c;
    float* proto;  /* [K,D] class prototypes on-ball */
    float* z;      /* [K] per-class scale */
    int step_hint;
} WubuHMLR;
int wubu_hmlr_init(WubuHMLR* m,int num_classes,int D,float c);
void wubu_hmlr_free(WubuHMLR* m);
float wubu_hmlr_distance(const float*a,const float*b,int D,float c);
void wubu_hmlr_logits(const WubuHMLR* m,const float* x,float* logits);
void wubu_hmlr_softmax(const float* logits,int K,float* probs);
float wubu_hmlr_train_step(WubuHMLR* m,const float* xs,const int* labels,
                           int N,int lr);
#ifdef __cplusplus
}
#endif
#endif

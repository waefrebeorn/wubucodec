/* GAP-D020: hyperbolic binary logistic regression */
#ifndef WUBU_HLOGREG_H
#define WUBU_HLOGREG_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int D;float c;
    float* proto;  /* decision prototype on-ball */
    float bias;
} WubuHLR;

int  wubu_hlr_init(WubuHLR* m,int D,float c,float bias_init);
void wubu_hlr_free(WubuHLR* m);
void wubu_hlr_logit(const WubuHLR* m,const float* x,float* logit);
float wubu_hlr_loss(WubuHLR* m,const float* xs,const int* labels,int n);
void wubu_hlr_train(WubuHLR* m,const float* xs,const int* labels,
                    int n,int iters,float lr);
int  wubu_hlr_predict(const WubuHLR* m,const float* x);
#ifdef __cplusplus
}
#endif
#endif

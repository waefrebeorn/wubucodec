/* GAP-C038: Riemannian Adam on the Poincaré ball */
#ifndef WUBU_RADAM_H
#define WUBU_RADAM_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int D,t;
    float lr,beta1,beta2;
    float* m;  /* first moment [D] */
    float* v;  /* second moment [D] */
} WubuRADAM;

int  wubu_radam_init(WubuRADAM* o,int D,float lr,float beta1,float beta2,
                     float eps_c);
void wubu_radam_free(WubuRADAM* o);
void wubu_radam_step(WubuRADAM* o,float* x,const float* egrad,float c);
void wubu_radam_batch(WubuRADAM* o,float* xs,int N,const float* grads,
                      int D,float c);
#ifdef __cplusplus
}
#endif
#endif

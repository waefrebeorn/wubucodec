/* GAP-C032: full hyperbolic transformer block */
#ifndef WUBU_HBLOCK_H
#define WUBU_HBLOCK_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_hblock_mobius_add(float* out,const float* u,const float* v,
                             int D,float c);
/* all matrices [D,D]; biases [D] or NULL. x/out: [N,D] on-ball. */
int wubu_hblock_forward(const float* W_att,const float* b_att,
                        const float* W_ff1,const float* b_ff1,
                        const float* W_ff2,const float* b_ff2,
                        const float* gamma,const float* beta,
                        const float* x,int N,int D,float c,float* out);
#ifdef __cplusplus
}
#endif
#endif

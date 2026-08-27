/* GAP-C031: Poincaré LayerNorm (tangent-space) */
#ifndef WUBU_HNORM_H
#define WUBU_HNORM_H
#ifdef __cplusplus
extern "C" {
#endif
/* x: [B,D] on-ball. gamma/beta: [D] or NULL (1/0). out: [B,D] on-ball. */
void wubu_hlayernorm(const float* x,int B,int D,float c,
                     const float* gamma,const float* beta,float* out);
#ifdef __cplusplus
}
#endif
#endif

/* GAP-D035: hyperbolic analogy completion */
#ifndef WUBU_HANALOGY_H
#define WUBU_HANALOGY_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_ha_log0(const float* x,int D,float c,float* out);
/* a:b :: c:? — returns nearest vocab index (excludes a,b,c), or -1. */
int wubu_ha_analogy(const float* emb,int vocab,int D,float c,
                     int a,int b,int cc,unsigned* seed);
#ifdef __cplusplus
}
#endif
#endif

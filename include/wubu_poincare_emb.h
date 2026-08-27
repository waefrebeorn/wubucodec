/* GAP-D017: Poincaré embeddings with negative sampling */
#ifndef WUBU_POINCARE_EMB_H
#define WUBU_POINCARE_EMB_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int n,D;float c,lr;
    float* emb; /* [n,D] on-ball */
} WubuPEmb;

int  wubu_pe_init(WubuPEmb* pe,int n_items,int D,float c,float lr,unsigned seed);
void wubu_pe_free(WubuPEmb* pe);
/* one training step on edge (u,v) with negatives; returns loss */
float wubu_pe_train_edge(WubuPEmb* pe,int u,int v,
                          const int* negatives,int n_neg);
float wubu_pe_dist_public(const WubuPEmb* pe,int u,int v);
#ifdef __cplusplus
}
#endif
#endif

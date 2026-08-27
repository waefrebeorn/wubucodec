/* GAP-D036: hyperbolic TransE knowledge graph embedding */
#ifndef WUBU_KGTRANS_H
#define WUBU_KGTRANS_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int n_ent,n_rel,D;
    float c;
    float* ent;   /* [n_ent, D] */
    float* rel;   /* [n_rel, D] tangent translations */
} WubuKG;

int  wubu_kg_init(WubuKG* kg,int n_ent,int n_rel,int D,float c,
                  unsigned seed);
void wubu_kg_free(WubuKG* kg);
float wubu_kg_score(const WubuKG* kg,int h,int r,int t);
float wubu_kg_train_epoch(WubuKG* kg,const int* heads,const int* rels,
                           const int* tails,int n,float margin,float lr,
                           unsigned* seed);
int  wubu_kg_rank(const WubuKG* kg,int h,int r,int t);
#ifdef __cplusplus
}
#endif
#endif

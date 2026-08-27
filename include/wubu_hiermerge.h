/* GAP-D016: hyperbolic hierarchical clustering (tree recovery) */
#ifndef WUBU_HIERMERGE_H
#define WUBU_HIERMERGE_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    float proto[64];  /* gyromidpoint prototype */
    int left,right;   /* children (-1 for leaf) */
    int size;
    int alive;
} WubuHMNode;

float wubu_hm_distance(const float* a,const float* b,int D,float c);
/* returns index of root in tree[2n-1]. */
int wubu_hm_cluster(const float* pts,int n,int D,float c,WubuHMNode* tree);
#ifdef __cplusplus
}
#endif
#endif

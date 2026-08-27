/* GAP-C047: hyperbolic hierarchical softmax */
#ifndef WUBU_HTREE_SOFTMAX_H
#define WUBU_HTREE_SOFTMAX_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int n_leaves,n_nodes,D;
    float c;
    int* left;int* right;
    float* proto;
} WubuHTS;

int  wubu_hts_init(WubuHTS* t,int n_leaves,int D,float c);
void wubu_hts_free(WubuHTS* t);
float wubu_hts_leaf_prob(const WubuHTS* t,const float* x,int leaf);
int  wubu_hts_predict(const WubuHTS* t,const float* x);
#ifdef __cplusplus
}
#endif
#endif

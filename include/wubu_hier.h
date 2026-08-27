/* GAP-D015: hyperbolic hierarchical classification */
#ifndef WUBU_HIER_H
#define WUBU_HIER_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int n,n_leaf,D;float c;
    int* parent;
    int* leaf_idx;
    float* proto; /* [n_nodes, D] from tree embedding */
} WubuHier;

int  wubu_hier_init(WubuHier* h,const int* parent,int n_nodes,
                    const int* leaf_indices,int n_leaves,int D,float c);
void wubu_hier_free(WubuHier* h);
int  wubu_hier_classify(const WubuHier* h,const float* x,int* out_conf);
float wubu_hier_accuracy(const WubuHier* h,const float* xs,
                          const int* true_leaf,int n);
#ifdef __cplusplus
}
#endif
#endif
extern int wubu_tree_embed(const int* parent,int n,int D,float tau,float* out);

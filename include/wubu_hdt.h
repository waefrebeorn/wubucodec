/* GAP-D024: hyperbolic decision tree */
#ifndef WUBU_HDT_H
#define WUBU_HDT_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    float anchor[64];  /* split anchor on-ball */
    float threshold;
    int left,right;    /* children (-1 for leaf) */
    int is_leaf,label;
} WubuHDTNode;

typedef struct {
    WubuHDTNode* nodes;
    int used,n,D;
    float c;
} WubuHDT;

int  wubu_hdt_build(WubuHDT* tree,const float* pts,const int* labels,
                    int n,int D,float c,int max_depth);
int  wubu_hdt_predict(const WubuHDT* tree,const float* x);
void wubu_hdt_free(WubuHDT* tree);
#ifdef __cplusplus
}
#endif
#endif

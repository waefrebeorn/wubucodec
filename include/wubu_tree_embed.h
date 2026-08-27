/* GAP-A021: Sarkar combinatorial tree embedding on Poincaré ball */
#ifndef WUBU_TREE_EMBED_H
#define WUBU_TREE_EMBED_H
#ifdef __cplusplus
extern "C" {
#endif
/* parent[i] = parent index (-1 for root); BFS order (parent[i]<i).
 * tau = hyperbolic edge length. out: [n, D]. Returns 0 on success. */
int wubu_tree_embed(const int* parent,int n,int D,float tau,float* out);
/* check max |d(i,parent(i)) - tau| across all edges. */
float wubu_tree_embed_check(const int* parent,const float* emb,
                             int n,int D,float c);
float tau_default(void);
#ifdef __cplusplus
}
#endif
#endif

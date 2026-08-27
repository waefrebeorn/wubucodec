/* GAP-D027: Dasgupta cost for hierarchical clustering trees */
#ifndef WUBU_DASGUPTA_H
#define WUBU_DASGUPTA_H
#ifdef __cplusplus
extern "C" {
#endif
/* left/right: [n_nodes] children (-1 = leaf); size: [n_nodes] subtree sizes;
 * w: [n_items,n_items] similarity weights. Returns Dasgupta cost. */
double wubu_dasgupta_cost(const int* left,const int* right,
                           const int* size,int n_nodes,
                           const float* w,int n_items);
#ifdef __cplusplus
}
#endif
#endif

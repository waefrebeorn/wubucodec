/* GAP-D010 companion: hyperbolic k-NN retrieval */
#ifndef WUBU_HKNN_H
#define WUBU_HKNN_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_hknn_distance(const float*a,const float*b,int D,float c);
/* brute-force top-k nearest by geodesic distance. out_idx[k], out_dist[k]. */
int wubu_hknn_search(const float* db,int n,const float* query,
                     int D,float c,int k,int* out_idx,float* out_dist);
/* batch recall@k with ground-truth labels, excluding self-match. */
float wubu_hknn_recall(const float* db_emb,const float* query_emb,
                       const int* gt_labels,int n,int D,float c,int k);
#ifdef __cplusplus
}
#endif
#endif

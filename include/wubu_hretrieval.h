/* GAP-D023: hierarchical retrieval scoring */
#ifndef WUBU_HRETRIEVAL_H
#define WUBU_HRETRIEVAL_H
#ifdef __cplusplus
extern "C" {
#endif
/* rank top-k db items by geodesic distance with optional hierarchy bonus */
int wubu_hr_rank(const float* db_emb,int n_db,const float* query,
                 int D,float c,const int* parent,const int* depth,
                 const int* db_label,int query_label,float hier_bonus,
                 float max_dist,int k,int* out_idx);
float wubu_hr_precision_k(const float* db_emb,int n,const float* queries,
                           const int* q_labels,const int* db_labels,
                           int n_q,int D,float c,int k);
#ifdef __cplusplus
}
#endif
#endif

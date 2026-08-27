/* GAP-D034: hyperbolic silhouette score */
#ifndef WUBU_SILHOUETTE_H
#define WUBU_SILHOUETTE_H
#ifdef __cplusplus
extern "C" {
#endif
/* pts: [n,D] on-ball; assign: [n] cluster ids in [0,k). Returns mean
 * silhouette in [-1,1], or -2 if undefined. */
float wubu_sil_score(const float* pts,const int* assign,
                      int n,int D,int k,float c);
#ifdef __cplusplus
}
#endif
#endif

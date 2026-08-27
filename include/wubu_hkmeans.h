/* GAP-C020: hyperbolic k-means on the Poincare ball */
#ifndef WUBU_HKMEANS_H
#define WUBU_HKMEANS_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_hkmeans_distance(const float* a,const float* b,int D,float c);
/* Full algorithm: assign + gyromidpoint update until convergence.
 * assignments: [n], centroids: [K,D]. Returns iterations used. */
int wubu_hkmeans(const float* pts,int n,int D,int K,float c,
                 int max_iter,float eps,
                 int* assignments,float* centroids);
#ifdef __cplusplus
}
#endif
#endif

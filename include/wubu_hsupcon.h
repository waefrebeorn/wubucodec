/* GAP-D031: hyperbolic supervised contrastive loss */
#ifndef WUBU_HSUPCON_H
#define WUBU_HSUPCON_H
#ifdef __cplusplus
extern "C" {
#endif
/* z: [N,D] on-ball; labels: [N]; label_sim: [n_labels,n_labels] optional
 * positive weighting (NULL = uniform). Returns mean SupCon loss. */
float wubu_hsc_loss(const float* z,const int* labels,
                     const float* label_sim,int n_labels,
                     int N,int D,float c,float tau);
#ifdef __cplusplus
}
#endif
#endif

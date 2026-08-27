/* GAP-B018: multi-hop geodesic neighborhood aggregation */
#ifndef WUBU_MULTIHOP_H
#define WUBU_MULTIHOP_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int wubu_mh_aggregate(const float* x,const int* adj_idx,
                       const int* adj_ptr,int n,int D,float c,
                       float tau,int max_hops,float* out);
#ifdef __cplusplus
}
#endif
#endif

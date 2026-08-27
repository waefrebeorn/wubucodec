/* Fast mode decision & CU depth pruning */
#ifndef WUBU_FASTMODE_H
#define WUBU_FASTMODE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int    wubu_fm_early_skip(long sad,long variance,long sad_threshold);
int    wubu_fm_should_split(double rd_parent,double rd_children_sum,
                              double lambda,int n_children);
double wubu_fm_estimate_split_cost(const uint8_t* orig,const uint8_t* pred,
                                     int W,int H,int bx,int by,int bs,double lambda);
int    wubu_fm_skip_intra(long sad_inter,long sad_intra_threshold);
int    wubu_aq_offset(const uint8_t* img,int W,int H,
                        int bx,int by,int bs,int max_offset);

typedef enum {FM_DECISION_SKIP,FM_DECISION_INTER_ONLY,FM_DECISION_FULL_SEARCH} FmDecision;
FmDecision wubu_fm_decide(long sad,long variance,long skip_threshold,long intra_threshold);
#ifdef __cplusplus
}
#endif
#endif

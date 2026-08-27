/* GAP-D022: graph-to-ball embedding via random walks */
#ifndef WUBU_GRAPH2BALL_H
#define WUBU_GRAPH2BALL_H
#include "wubu_poincare_emb.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int n,D;float c,lr;
    WubuPEmb pe;
} WubuG2B;

int  wubu_g2b_init(WubuG2B* g,int n,int D,float c,float lr,unsigned seed);
void wubu_g2b_free(WubuG2B* g);
void wubu_g2b_walk(const int* adj_idx,const int* adj_ptr,
                   int n,int start,int len,int* walk);
float wubu_g2b_train(WubuG2B* g,const int* adj_idx,const int* adj_ptr,
                     int walk_len,int num_walks,int window,int k_neg);
float wubu_g2b_separation(const WubuG2B* g,const int* adj_idx,
                           const int* adj_ptr);
#ifdef __cplusplus
}
#endif
#endif

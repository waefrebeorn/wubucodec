/* GAP-B017: attention entropy monitor */
#ifndef WUBU_ATTNENTROPY_H
#define WUBU_ATTNENTROPY_H
#ifdef __cplusplus
extern "C" {
#endif
float wubu_ae_row_entropy(const float* row,int N);
float wubu_ae_mean_entropy(const float* attn,int N,int n_heads);
int  wubu_ae_collapsed(const float* attn,int N,int n_heads,float thresh);
#ifdef __cplusplus
}
#endif
#endif

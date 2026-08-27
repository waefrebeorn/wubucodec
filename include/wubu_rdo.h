/* GROUP 14: Rate-Distortion Optimization */
#ifndef WUBU_RDO_H
#define WUBU_RDO_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {WUBU_I_FRAME,WUBU_P_FRAME,WUBU_B_FRAME} wubu_frame_type_t;
typedef enum {WUBU_MODE_SKIP,WUBU_MODE_INTER,WUBU_MODE_INTRA} wubu_mode_t;

double wubu_lambda_from_qp(int qp,wubu_frame_type_t frame_type);
long   wubu_rd_sse(const uint8_t* orig,const uint8_t* recon,long n);
int    wubu_rd_estimate_bits(const int16_t* quantized_coeffs,int n_coeffs);
double wubu_rd_cost(long sse,int estimated_bits,double lambda);
wubu_mode_t wubu_best_mode(const uint8_t* orig,const uint8_t* inter_pred,
                             const uint8_t* intra_pred,
                             const int16_t* residual_coeffs,
                             int n_pixels,int n_coeffs,double lambda);
int    wubu_can_early_terminate_skip(long sad,long block_variance,long threshold);
int    wubu_hier_qp_offset(int temporal_layer);
#ifdef __cplusplus
}
#endif
#endif

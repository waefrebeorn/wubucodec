/* wubu_encoder.h — C15 keystone: unified RDO encoder loop API */
#ifndef WUBU_ENCODER_H
#define WUBU_ENCODER_H
#include "wubu_rdo.h"
#include "wubu_transform.h"
#include "wubu_bframe2.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

int wubu_encode_frame(const uint8_t* orig,
                       const uint8_t* ref_past,
                       const uint8_t* ref_future,
                       int fw,int fh,int qp,
                       wubu_frame_type_t frame_type,
                       const int16_t* mv_grid,
                       uint8_t* recon,
                       long* total_bits);

typedef struct {
    int    qp;
    long   bits;
    double psnr;
} RdCurvePoint;

int wubu_rd_curve(const uint8_t* orig,
                   const uint8_t* ref_past,
                   const uint8_t* ref_future,
                   int fw,int fh,
                   const int* qp_vals,int n_qp,
                   wubu_frame_type_t frame_type,
                   const int16_t* mv_grid,
                   RdCurvePoint* curve);

#ifdef __cplusplus
}
#endif
#endif

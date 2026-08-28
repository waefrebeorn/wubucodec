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

/* C7: Encode a GOP-8 with hierarchical B-frame structure */
int wubu_encode_gop8(const uint8_t* frames,int W,int H,int base_qp,
                      uint8_t* recon_frames,long* total_bits);

/* Encode with preset (0=ultrafast, 1=fast, 2=medium, 3=slow) */
int wubu_encode_frame_preset(const uint8_t* orig,
                              const uint8_t* ref_past,
                              const uint8_t* ref_future,
                              int W, int H, int qp,
                              wubu_frame_type_t frame_type,
                              const int16_t* mv_grid,
                              uint8_t* recon_out,
                              long* total_bits,
                              int preset_idx);

#ifdef __cplusplus
}
#endif
#endif

/* wubu_wpp.h — Wavefront Parallel Processing for within-frame parallelism */
#ifndef WUBU_WPP_H
#define WUBU_WPP_H
#include <stdint.h>
#include "wubu_encoder.h"

/* Encode frame with WPP row parallelism (row-per-thread) */
int wubu_encode_frame_wpp(const uint8_t* orig, const uint8_t* ref,
                           int W, int H, int qp,
                           uint8_t* recon, long* total_bits,
                           int num_threads, int preset);

#endif

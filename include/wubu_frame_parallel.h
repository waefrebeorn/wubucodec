/* wubu_frame_parallel.h — frame-level parallel encoding */
#ifndef WUBU_FRAME_PARALLEL_H
#define WUBU_FRAME_PARALLEL_H
#include <stdint.h>
#include "wubu_encoder.h"

/* Encode I-frames in parallel (no temporal dependencies) */
int wubu_encode_i_frames_parallel(const uint8_t* frames, int W, int H, int qp,
                                   int start_frame, int end_frame,
                                   uint8_t* recons, long* total_bits,
                                   int num_threads);

#endif

/* wubu_tile_parallel.h — tile-based parallel encoding for 8K */
#ifndef WUBU_TILE_PARALLEL_H
#define WUBU_TILE_PARALLEL_H
#include <stdint.h>
#include "wubu_encoder.h"

/* Encode frame in tiles using multiple threads */
int wubu_encode_frame_tiled(const uint8_t* orig, const uint8_t* ref,
                             int W, int H, int qp,
                             uint8_t* recon, long* total_bits,
                             int num_tiles, int preset);

#endif

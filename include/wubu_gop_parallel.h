/* wubu_gop_parallel.h — GOP-level parallelism for multi-GOP sequences */
#ifndef WUBU_GOP_PARALLEL_H
#define WUBU_GOP_PARALLEL_H
#include <stdint.h>

typedef struct {
    const uint8_t* frames;
    uint8_t* recons;
    int W, H, qp;
    int gop_start;
    int gop_size;
    long* total_bits;
} GopJob;

/* Encode a sequence of frames split into GOPs */
int wubu_encode_sequence_parallel(const uint8_t* frames, int W, int H, int qp,
                                   int total_frames, int gop_size,
                                   uint8_t* recons, long* total_bits,
                                   int num_threads);

#endif

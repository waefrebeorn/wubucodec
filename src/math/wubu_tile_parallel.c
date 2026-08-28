/* wubu_tile_parallel.c — tile-based parallel encoding for 8K
 *
 * Split frame into horizontal strips (tiles) and encode each in parallel.
 * Each tile is encoded as a P-frame using the strip above as reference.
 * This avoids the I-frame overhead and gives proper inter-frame prediction.
 */

#include "wubu_tile_parallel.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct {
    const uint8_t* orig;
    const uint8_t* ref;
    uint8_t* recon;
    int W, H, qp;
    int y_start, y_end;  /* tile boundaries in pixels */
    int preset;
    long* total_bits;
} TileJob;

static void* encode_tile(void* arg){
    TileJob* job = (TileJob*)arg;
    int tile_h = job->y_end - job->y_start;
    int np = job->W * tile_h;
    int y_offset = job->y_start * job->W;

    /* Encode this tile as a P-frame using the strip above as reference */
    long bits = 0;
    wubu_encode_frame_preset(job->orig + y_offset, job->ref + y_offset, NULL,
                              job->W, tile_h, job->qp,
                              WUBU_P_FRAME, NULL, job->recon + y_offset, &bits, job->preset);
    *job->total_bits = bits;
    return NULL;
}

/* Encode frame in tiles using multiple threads */
int wubu_encode_frame_tiled(const uint8_t* orig, const uint8_t* ref,
                             int W, int H, int qp,
                             uint8_t* recon, long* total_bits,
                             int num_tiles, int preset){
    if(num_tiles <= 1 || H < num_tiles * 8){
        /* Single-threaded fallback */
        *total_bits = 0;
        long bits;
        wubu_encode_frame_preset(orig, ref, NULL, W, H, qp, WUBU_I_FRAME, NULL, recon, &bits, preset);
        *total_bits = bits;
        return 0;
    }

    /* Limit tiles to reasonable count */
    if(num_tiles > 8) num_tiles = 8;

    TileJob* jobs = malloc(sizeof(TileJob) * num_tiles);
    pthread_t* threads = malloc(sizeof(pthread_t) * num_tiles);
    long* tile_bits = calloc(num_tiles, sizeof(long));

    int rows_per_tile = H / num_tiles;
    int extra = H % num_tiles;

    int y = 0;
    for(int t = 0; t < num_tiles; t++){
        jobs[t].orig = orig;
        jobs[t].ref = ref;
        jobs[t].recon = recon;
        jobs[t].W = W;
        jobs[t].H = H;
        jobs[t].qp = qp;
        jobs[t].y_start = y;
        jobs[t].y_end = y + rows_per_tile + (t < extra ? 1 : 0);
        jobs[t].preset = preset;
        jobs[t].total_bits = &tile_bits[t];
        y = jobs[t].y_end;
        pthread_create(&threads[t], NULL, encode_tile, &jobs[t]);
    }

    *total_bits = 0;
    for(int t = 0; t < num_tiles; t++){
        pthread_join(threads[t], NULL);
        *total_bits += tile_bits[t];
    }

    free(jobs);
    free(threads);
    free(tile_bits);
    return 0;
}

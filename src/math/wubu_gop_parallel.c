/* wubu_gop_parallel.c — GOP-level parallelism for multi-GOP sequences
 *
 * Split a sequence into independent GOPs and encode each GOP in a separate thread.
 * GOP boundaries are I-frames, so no temporal dependencies between GOPs.
 * This gives perfect parallelism for long sequences.
 */

#include "wubu_gop_parallel.h"
#include "wubu_encoder.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* Thread function for encoding a range of GOPs */
static void* encode_gop_range(void* arg) {
    GopJob* job = (GopJob*)arg;
    int np = job->W * job->H;
    long bits = 0;

    /* Encode frames sequentially in this thread's range */
    long frame_bits = 0;
    /* First frame is I-frame */
    wubu_encode_frame_preset(job->frames + job->gop_start * np, NULL, NULL,
                              job->W, job->H, job->qp,
                              WUBU_I_FRAME, NULL,
                              job->recons + job->gop_start * np, &frame_bits, 1);
    bits += frame_bits;

    /* Rest are P-frames */
    for (int f = 1; f < job->gop_size; f++) {
        int idx = job->gop_start + f;
        frame_bits = 0;
        wubu_encode_frame_preset(job->frames + idx * np,
                                  job->recons + (idx - 1) * np, NULL,
                                  job->W, job->H, job->qp,
                                  WUBU_P_FRAME, NULL,
                                  job->recons + idx * np, &frame_bits, 1);
        bits += frame_bits;
    }

    *job->total_bits = bits;
    return NULL;
}

/* Encode a sequence of frames split into GOPs */
int wubu_encode_sequence_parallel(const uint8_t* frames, int W, int H, int qp,
                                   int total_frames, int gop_size,
                                   uint8_t* recons, long* total_bits,
                                   int num_threads) {
    if (num_threads <= 1 || gop_size >= total_frames) {
        /* Single-threaded fallback */
        *total_bits = 0;
        int np = W * H;
        long frame_bits = 0;
        wubu_encode_frame_preset(frames, NULL, NULL, W, H, qp,
                                  WUBU_I_FRAME, NULL, recons, &frame_bits, 1);
        *total_bits += frame_bits;
        for (int f = 1; f < total_frames; f++) {
            frame_bits = 0;
            wubu_encode_frame_preset(frames + f * np, recons + (f-1) * np, NULL,
                                      W, H, qp, WUBU_P_FRAME, NULL,
                                      recons + f * np, &frame_bits, 1);
            *total_bits += frame_bits;
        }
        return 0;
    }

    /* Calculate number of GOPs */
    int n_gops = (total_frames + gop_size - 1) / gop_size;
    if (num_threads > n_gops) num_threads = n_gops;

    GopJob* jobs = malloc(sizeof(GopJob) * num_threads);
    pthread_t* threads = malloc(sizeof(pthread_t) * num_threads);
    long* gop_bits = calloc(num_threads, sizeof(long));

    /* Distribute GOPs across threads (contiguous ranges for cache locality) */
    int gop_per_thread = n_gops / num_threads;
    int extra = n_gops % num_threads;

    int gop = 0;
    for (int t = 0; t < num_threads; t++) {
        jobs[t].frames = frames;
        jobs[t].recons = recons;
        jobs[t].W = W;
        jobs[t].H = H;
        jobs[t].qp = qp;
        jobs[t].total_bits = &gop_bits[t];

        /* Each thread gets a contiguous range of GOPs */
        int start_gop = gop;
        int n_local_gops = gop_per_thread + (t < extra ? 1 : 0);
        gop += n_local_gops;

        /* Encode this thread's GOPs sequentially */
        jobs[t].gop_start = start_gop * gop_size;
        int end_frame = (start_gop + n_local_gops) * gop_size;
        if (end_frame > total_frames) end_frame = total_frames;
        jobs[t].gop_size = end_frame - jobs[t].gop_start;

        pthread_create(&threads[t], NULL, encode_gop_range, &jobs[t]);
    }

    *total_bits = 0;
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
        *total_bits += gop_bits[t];
    }

    free(jobs);
    free(threads);
    free(gop_bits);
    return 0;
}

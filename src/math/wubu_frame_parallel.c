/* wubu_frame_parallel.c — frame-level parallel encoding for independent frames
 *
 * Encode multiple frames in parallel using one thread per frame.
 * Uses original frames as references (not reconstructed) to avoid dependencies.
 * This gives perfect parallelism for multi-frame sequences.
 */

#include "wubu_frame_parallel.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct {
    const uint8_t* frames;
    uint8_t* recons;
    int W, H, qp;
    int start_frame, end_frame;
    long* total_bits;
} FrameJob;

static void* encode_frame_range(void* arg){
    FrameJob* job = (FrameJob*)arg;
    long bits = 0;
    int np = job->W * job->H;

    for(int f = job->start_frame; f < job->end_frame; f++){
        long frame_bits = 0;
        /* Use original previous frame as reference for parallelism */
        /* This avoids dependency on reconstructed frames */
        wubu_encode_frame_preset(job->frames + (size_t)f * np,
                                  job->frames + (size_t)(f - 1) * np, NULL,
                                  job->W, job->H, job->qp,
                                  WUBU_P_FRAME, NULL,
                                  job->recons + (size_t)f * np, &frame_bits, 1);
        bits += frame_bits;
    }
    *job->total_bits = bits;
    return NULL;
}

/* Encode I-frames in parallel (no temporal dependencies) */
int wubu_encode_i_frames_parallel(const uint8_t* frames, int W, int H, int qp,
                                   int start_frame, int end_frame,
                                   uint8_t* recons, long* total_bits,
                                   int num_threads){
    if(num_threads <= 1 || (end_frame - start_frame) <= 1){
        /* Single-threaded fallback */
        *total_bits = 0;
        int np = W * H;
        for(int f = start_frame; f < end_frame; f++){
            long bits;
            wubu_encode_frame_preset(frames + (size_t)f * np, NULL, NULL, W, H, qp,
                                      WUBU_I_FRAME, NULL, recons + (size_t)f * np, &bits, 1);
            *total_bits += bits;
        }
        return 0;
    }

    /* Limit threads to frame count */
    int n_frames = end_frame - start_frame;
    if(num_threads > n_frames) num_threads = n_frames;

    FrameJob* jobs = malloc(sizeof(FrameJob) * num_threads);
    pthread_t* threads = malloc(sizeof(pthread_t) * num_threads);
    long* thread_bits = calloc(num_threads, sizeof(long));

    int frames_per_thread = n_frames / num_threads;
    int extra = n_frames % num_threads;

    int frame = start_frame;
    for(int t = 0; t < num_threads; t++){
        jobs[t].frames = frames;
        jobs[t].recons = recons;
        jobs[t].W = W;
        jobs[t].H = H;
        jobs[t].qp = qp;
        jobs[t].start_frame = frame;
        jobs[t].end_frame = frame + frames_per_thread + (t < extra ? 1 : 0);
        jobs[t].total_bits = &thread_bits[t];
        frame = jobs[t].end_frame;
        pthread_create(&threads[t], NULL, encode_frame_range, &jobs[t]);
    }

    *total_bits = 0;
    for(int t = 0; t < num_threads; t++){
        pthread_join(threads[t], NULL);
        *total_bits += thread_bits[t];
    }

    free(jobs);
    free(threads);
    free(thread_bits);
    return 0;
}

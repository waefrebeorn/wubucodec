/* wubu_parallel_bench.c — measure multi-threaded encoding at 1080p
 * Uses frame-level parallelism: each thread encodes different frames.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "wubu_encoder.h"

#define FRAME_W 1920
#define FRAME_H 1080
#define NF 60

typedef struct {
    const uint8_t* frames;
    uint8_t* recons;
    int width, height, qp;
    int start_frame, end_frame;
    long* total_bits;
} FrameJob;

static void* encode_frame_range(void* arg){
    FrameJob* job = (FrameJob*)arg;
    long bits = 0;
    int np = job->width * job->height;

    for(int f = job->start_frame; f < job->end_frame; f++){
        long frame_bits = 0;
        /* Use original previous frame as reference */
        wubu_encode_frame_preset(job->frames + (size_t)f * np,
                                  job->frames + (size_t)(f - 1) * np, NULL,
                                  job->width, job->height, job->qp,
                                  WUBU_P_FRAME, NULL,
                                  job->recons + (size_t)f * np, &frame_bits, 1);
        bits += frame_bits;
    }
    *job->total_bits = bits;
    return NULL;
}

int main(void){
    const int np = FRAME_W * FRAME_H;
    uint8_t* frames = malloc((size_t)NF * np);
    uint8_t* recons = malloc((size_t)NF * np);

    for(int f=0; f<NF; f++)
        for(int y=0; y<FRAME_H; y++)
            for(int x=0; x<FRAME_W; x++){
                float edge = 960.0f + f * 0.7f;
                float v = (x < edge) ? 100.0f : 200.0f;
                v += 20.0f * sinf(2.0f * 3.14159f * (x + f) / FRAME_W) * cosf(2.0f * 3.14159f * y / FRAME_H);
                if(v<0)v=0; if(v>255)v=255;
                frames[(size_t)f*np + y*FRAME_W + x] = (uint8_t)v;
            }

    printf("=== WuBuCodec Parallel Benchmark (1080p) ===\n");
    printf("Content: %dx%d, %d frames\n\n", FRAME_W, FRAME_H, NF);

    /* Single-threaded baseline */
    {
        int qp = 26;
        clock_t start = clock();
        long total_bits = 0, frame_bits = 0;
        wubu_encode_frame_preset(frames, NULL, NULL, FRAME_W, FRAME_H, qp,
                                  WUBU_I_FRAME, NULL, recons, &frame_bits, 1);
        total_bits += frame_bits;
        for(int f=1; f<NF; f++){
            frame_bits = 0;
            wubu_encode_frame_preset(frames + (size_t)f * np, recons + (size_t)(f-1)*np, NULL,
                                      FRAME_W, FRAME_H, qp, WUBU_P_FRAME, NULL,
                                      recons + (size_t)f * np, &frame_bits, 1);
            total_bits += frame_bits;
        }
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Single-threaded: %6.1f fps, %7.1f kbps\n", NF / elapsed, total_bits * 30.0 / NF / 1024.0);
    }

    /* Frame-parallel: each thread encodes different frames */
    for(int n_threads = 2; n_threads <= 4; n_threads++){
        int qp = 26;
        clock_t start = clock();

        /* Encode I-frame first */
        long total_bits = 0, frame_bits = 0;
        wubu_encode_frame_preset(frames, NULL, NULL, FRAME_W, FRAME_H, qp,
                                  WUBU_I_FRAME, NULL, recons, &frame_bits, 1);
        total_bits += frame_bits;

        /* Encode remaining frames in parallel */
        int n_remaining = NF - 1;
        int nt = n_threads;
        if (nt > n_remaining) nt = n_remaining;

        FrameJob* jobs = malloc(sizeof(FrameJob) * nt);
        pthread_t* threads = malloc(sizeof(pthread_t) * nt);
        long* thread_bits = calloc(nt, sizeof(long));

        int frames_per_thread = n_remaining / nt;
        int extra = n_remaining % nt;

        int frame = 1;
        for(int t = 0; t < nt; t++){
            jobs[t].frames = frames;
            jobs[t].recons = recons;
            jobs[t].width = FRAME_W;
            jobs[t].height = FRAME_H;
            jobs[t].qp = qp;
            jobs[t].start_frame = frame;
            jobs[t].end_frame = frame + frames_per_thread + (t < extra ? 1 : 0);
            jobs[t].total_bits = &thread_bits[t];
            frame = jobs[t].end_frame;
            pthread_create(&threads[t], NULL, encode_frame_range, &jobs[t]);
        }

        for(int t = 0; t < nt; t++){
            pthread_join(threads[t], NULL);
            total_bits += thread_bits[t];
        }

        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("%d-thread frame-parallel: %6.1f fps, %7.1f kbps\n", nt, NF / elapsed, total_bits * 30.0 / NF / 1024.0);

        free(jobs);
        free(threads);
        free(thread_bits);
    }

    free(frames);
    free(recons);
    return 0;
}

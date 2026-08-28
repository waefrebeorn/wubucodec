/* wubu_thread_debug.c — debug thread pool crash */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "wubu_encoder.h"
#include "wubu_thread_pool.h"

#define W 176
#define H 144
#define NF 8

int main(void){
    const int np = W*H;
    uint8_t* frames = malloc((size_t)NF * np);
    uint8_t* recons = malloc((size_t)NF * np);

    for(int f=0; f<NF; f++)
        for(int i=0; i<np; i++)
            frames[(size_t)f*np + i] = (uint8_t)((f + i) % 256);

    printf("Testing thread pool with %dx%d, %d frames\n", W, H, NF);

    /* Single-threaded first */
    printf("Single-threaded...\n");
    long total_bits = 0, frame_bits = 0;
    wubu_encode_frame_preset(frames, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recons, &frame_bits, 1);
    total_bits += frame_bits;
    for(int f=1; f<NF; f++){
        frame_bits = 0;
        wubu_encode_frame_preset(frames + f*np, recons + (f-1)*np, NULL, W, H, 26, WUBU_P_FRAME, NULL, recons + f*np, &frame_bits, 1);
        total_bits += frame_bits;
    }
    printf("OK: %ld bits\n", total_bits);

    /* Thread pool */
    printf("Thread pool (2 threads)...\n");
    wubu_thread_pool_init(2);
    total_bits = 0;
    wubu_encode_frame_pooled(frames, NULL, W, H, 26, WUBU_I_FRAME, recons, &frame_bits, 1);
    total_bits += frame_bits;
    for(int f=1; f<NF; f++){
        frame_bits = 0;
        wubu_encode_frame_pooled(frames + f*np, recons + (f-1)*np, W, H, 26, WUBU_P_FRAME, recons + f*np, &frame_bits, 1);
        total_bits += frame_bits;
    }
    wubu_thread_pool_shutdown();
    printf("OK: %ld bits\n", total_bits);

    free(frames);
    free(recons);
    return 0;
}

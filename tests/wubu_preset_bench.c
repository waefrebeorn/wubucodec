/* wubu_preset_bench.c — benchmark all encoder presets with speed/quality tradeoff */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "wubu_encoder.h"

#define FRAME_W 1920
#define FRAME_H 1080
#define NF 30

int main(void){
    const int np = FRAME_W * FRAME_H;
    uint8_t* frames = malloc((size_t)NF * np);
    uint8_t* recons = malloc((size_t)NF * np);

    /* Generate test content */
    for(int f=0; f<NF; f++)
        for(int y=0; y<FRAME_H; y++)
            for(int x=0; x<FRAME_W; x++){
                float edge = 960.0f + f * 0.7f;
                float v = (x < edge) ? 100.0f : 200.0f;
                v += 20.0f * sinf(2.0f * 3.14159f * (x + f) / FRAME_W) * cosf(2.0f * 3.14159f * y / FRAME_H);
                if(v<0)v=0; if(v>255)v=255;
                frames[(size_t)f*np + y*FRAME_W + x] = (uint8_t)v;
            }

    const char* preset_names[] = {"superfast", "ultrafast", "fast", "medium", "slow"};
    int qps[] = {20, 26, 32, 40};

    printf("=== WuBuCodec Preset Benchmark (1080p) ===\n");
    printf("Content: %dx%d, %d frames\n\n", FRAME_W, FRAME_H, NF);

    printf("%-10s %6s %8s %10s %8s\n", "Preset", "QP", "fps", "kbps", "PSNR");
    printf("%-10s %6s %8s %10s %8s\n", "------", "--", "---", "----", "----");

    for(int preset = 0; preset < 5; preset++){
        for(int q = 0; q < 4; q++){
            int qp = qps[q];
            clock_t start = clock();
            long total_bits = 0, frame_bits = 0;

            wubu_encode_frame_preset(frames, NULL, NULL, FRAME_W, FRAME_H, qp,
                                      WUBU_I_FRAME, NULL, recons, &frame_bits, preset);
            total_bits += frame_bits;

            for(int f=1; f<NF; f++){
                frame_bits = 0;
                wubu_encode_frame_preset(frames + (size_t)f * np,
                                          recons + (size_t)(f-1) * np, NULL,
                                          FRAME_W, FRAME_H, qp, WUBU_P_FRAME, NULL,
                                          recons + (size_t)f * np, &frame_bits, preset);
                total_bits += frame_bits;
            }

            clock_t end = clock();
            double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
            double fps = NF / elapsed;

            /* Measure PSNR */
            long total_mse = 0;
            for(int f=0; f<NF; f++){
                long frame_bits = 0;
                wubu_encode_frame_preset(frames + (size_t)f * np,
                                          f > 0 ? recons + (size_t)(f-1) * np : NULL, NULL,
                                          FRAME_W, FRAME_H, qp,
                                          f == 0 ? WUBU_I_FRAME : WUBU_P_FRAME, NULL,
                                          recons + (size_t)f * np, &frame_bits, preset);
                for(int p=0; p<np; p++){
                    int d = frames[(size_t)f*np + p] - recons[(size_t)f*np + p];
                    total_mse += d*d;
                }
            }
            double avg_psnr = total_mse > 0 ? 10.0 * log10(255.0*255.0 * NF * np / (double)total_mse) : 99.0;

            printf("%-10s %6d %8.1f %10.1f %8.1f\n", preset_names[preset], qp, fps, total_bits * 30.0 / NF / 1024.0, avg_psnr);
        }
        printf("\n");
    }

    free(frames);
    free(recons);
    return 0;
}

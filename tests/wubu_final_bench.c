/* Final benchmark: 1080p + QCIF */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "wubu_encoder.h"

int main(void){
    const char* preset_names[] = {"superfast", "ultrafast", "fast", "medium", "slow"};
    
    /* 1080p benchmark */
    printf("=== WuBuCodec Final Benchmark (1080p) ===\n");
    int W = 1920, H = 1080, np = W*H;
    uint8_t* frames = malloc((size_t)30 * np);
    uint8_t* recons = malloc((size_t)30 * np);
    for(int i=0; i<30*np; i++) frames[i] = (uint8_t)((i * 13) % 256);
    
    for(int preset = 0; preset < 5; preset++){
        clock_t start = clock();
        long total_bits = 0;
        for(int f = 0; f < 30; f++){
            long bits = 0;
            wubu_encode_frame_preset(frames + (size_t)f * np, 
                                      f > 0 ? recons + (size_t)(f-1) * np : NULL, NULL,
                                      W, H, 26, f == 0 ? WUBU_I_FRAME : WUBU_P_FRAME, NULL,
                                      recons + (size_t)f * np, &bits, preset);
            total_bits += bits;
        }
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("%-10s: %6.1f fps, %7.1f kbps\n", preset_names[preset], 30.0 / elapsed, total_bits * 30.0 / 30 / 1024.0);
    }
    
    free(frames);
    free(recons);
    
    /* QCIF benchmark */
    printf("\n=== QCIF (176x144) ===\n");
    W = 176; H = 144; np = W*H;
    frames = malloc((size_t)30 * np);
    recons = malloc((size_t)30 * np);
    for(int i=0; i<30*np; i++) frames[i] = (uint8_t)(i % 256);
    
    for(int preset = 0; preset < 5; preset++){
        clock_t start = clock();
        for(int f = 0; f < 30; f++){
            long bits = 0;
            wubu_encode_frame_preset(frames + (size_t)f * np, 
                                      f > 0 ? recons + (size_t)(f-1) * np : NULL, NULL,
                                      W, H, 26, f == 0 ? WUBU_I_FRAME : WUBU_P_FRAME, NULL,
                                      recons + (size_t)f * np, &bits, preset);
        }
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        printf("%-10s: %6.1f fps\n", preset_names[preset], 30.0 / elapsed);
    }
    
    free(frames);
    free(recons);
    return 0;
}

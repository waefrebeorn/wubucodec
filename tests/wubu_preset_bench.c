/* wubu_preset_bench.c — benchmark all encoder presets */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "wubu_encoder.h"

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

#define W 176
#define H 144
#define NF 30

static double compute_psnr(const uint8_t* orig, const uint8_t* recon, int n) {
    long mse = 0;
    for (int i = 0; i < n; i++) { int d = orig[i] - recon[i]; mse += d * d; }
    double mse_d = (double)mse / n;
    return mse_d > 0 ? 10.0 * log10(255.0 * 255.0 / mse_d) : 99.0;
}

int main(void){
    const int np = W*H;
    uint8_t* frames = malloc((size_t)NF * np);
    
    /* Generate test content: motion + detail */
    for(int f=0; f<NF; f++){
        for(int y=0; y<H; y++){
            for(int x=0; x<W; x++){
                float edge = 80.0f + f * 0.7f;
                float v = (x < edge) ? 100.0f : 200.0f;
                v += 20.0f * sinf(2.0f * M_PI * (x + f) / W) * cosf(2.0f * M_PI * y / H);
                v += 10.0f * ((x + y + f) % 17);
                if(v<0)v=0; if(v>255)v=255;
                frames[(size_t)f*np + y*W + x] = (uint8_t)v;
            }
        }
    }
    
    const char* preset_names[] = {"superfast", "ultrafast", "fast", "medium", "slow"};
    
    printf("=== WuBuCodec Preset Benchmark ===\n");
    printf("Content: %dx%d, %d frames, motion+texture\n\n", W, H, NF);
    
    /* P-frame benchmark across presets */
    printf("%-10s %8s %10s %8s %10s\n", "Preset", "fps", "kbps", "PSNR", "bits/frame");
    printf("%-10s %8s %10s %8s %10s\n", "------", "---", "----", "----", "-----------");
    
    for(int preset = 0; preset < 5; preset++){
        int qp = 26;
        uint8_t* recon = malloc(np);
        uint8_t* ref = malloc(np);
        clock_t start = clock();
        long total_bits = 0;
        long bits;
        
        wubu_encode_frame_preset(frames, NULL, NULL, W, H, qp, WUBU_I_FRAME, NULL, ref, &bits, preset);
        total_bits = bits;
        
        for(int f=1; f<NF; f++){
            bits = 0;
            wubu_encode_frame_preset(frames + f*np, ref, NULL, W, H, qp, WUBU_P_FRAME, NULL, recon, &bits, preset);
            total_bits += bits;
            memcpy(ref, recon, np);
        }
        
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double fps = NF / elapsed;
        
        /* Measure PSNR */
        long total_mse = 0;
        wubu_encode_frame_preset(frames, NULL, NULL, W, H, qp, WUBU_I_FRAME, NULL, ref, &bits, preset);
        for(int f=1; f<NF; f++){
            bits = 0;
            wubu_encode_frame_preset(frames + f*np, ref, NULL, W, H, qp, WUBU_P_FRAME, NULL, recon, &bits, preset);
            for(int p=0; p<np; p++){
                int d = frames[f*np+p] - recon[p];
                total_mse += d*d;
            }
            memcpy(ref, recon, np);
        }
        double avg_psnr = total_mse > 0 ? 10.0 * log10(255.0*255.0 * (NF-1) * np / (double)total_mse) : 99.0;
        
        printf("%-10s %8.1f %10.1f %8.1f %10ld\n", preset_names[preset], fps, total_bits * 30.0 / NF / 1024.0, avg_psnr, total_bits/NF);
        
        free(recon);
        free(ref);
    }
    
    /* I-frame benchmark across presets */
    printf("\n=== I-frame only ===\n");
    for(int preset = 0; preset < 5; preset++){
        int qp = 26;
        uint8_t* recon = malloc(np);
        clock_t start = clock();
        long total_bits = 0;
        
        for(int f=0; f<NF; f++){
            long bits;
            wubu_encode_frame_preset(frames + f*np, NULL, NULL, W, H, qp, WUBU_I_FRAME, NULL, recon, &bits, preset);
            total_bits += bits;
        }
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double fps = NF / elapsed;
        
        printf("%-10s: %6.1f fps\n", preset_names[preset], fps);
        free(recon);
    }
    
    free(frames);
    return 0;
}

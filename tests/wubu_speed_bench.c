/* wubu_speed_bench.c — measure encode speed for different presets */
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
    
    /* Generate realistic test content: motion + detail */
    for(int f=0; f<NF; f++){
        for(int y=0; y<H; y++){
            for(int x=0; x<W; x++){
                /* Moving gradient + sharp edge + texture */
                float edge = 80.0f + f * 0.7f;
                float v = (x < edge) ? 100.0f : 200.0f;
                v += 20.0f * sinf(2.0f * M_PI * (x + f) / W) * cosf(2.0f * M_PI * y / H);
                v += 10.0f * ((x + y + f) % 17);
                if(v<0)v=0; if(v>255)v=255;
                frames[(size_t)f*np + y*W + x] = (uint8_t)v;
            }
        }
    }
    
    printf("=== WuBuCodec Speed Benchmark ===\n");
    printf("Content: %dx%d, %d frames, motion+texture\n\n", W, H, NF);
    
    int qps[] = {20, 26, 32, 40};
    const char* preset_names[] = {"I-frame only", "P-frame (ME)", "B-frame (bi-pred)"};
    
    /* I-frame benchmark */
    {
        int qp = 26;
        uint8_t* recon = malloc(np);
        clock_t start = clock();
        long total_bits = 0;
        for(int f=0; f<NF; f++){
            long bits = 0;
            wubu_encode_frame(frames + f*np, NULL, NULL, W, H, qp, WUBU_I_FRAME, NULL, recon, &bits);
            total_bits += bits;
        }
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double fps = NF / elapsed;
        double avg_psnr = 0;
        for(int f=0; f<NF; f++){
            uint8_t* recon = malloc(np);
            long bits;
            wubu_encode_frame(frames + f*np, NULL, NULL, W, H, qp, WUBU_I_FRAME, NULL, recon, &bits);
            avg_psnr += compute_psnr(frames + f*np, recon, np);
            free(recon);
        }
        avg_psnr /= NF;
        printf("%-20s QP=%d: %6.1f fps, %7.1f kbps, %5.1f dB, %ld bits/frame\n",
               preset_names[0], qp, fps, total_bits * 30.0 / NF / 1024.0, avg_psnr, total_bits/NF);
        free(recon);
    }
    
    /* P-frame benchmark */
    {
        int qp = 26;
        uint8_t* recon = malloc(np);
        uint8_t* ref = malloc(np);
        clock_t start = clock();
        long total_bits = 0;
        /* First frame is I */
        long bits;
        wubu_encode_frame(frames, NULL, NULL, W, H, qp, WUBU_I_FRAME, NULL, ref, &bits);
        total_bits = bits;
        /* Rest are P */
        for(int f=1; f<NF; f++){
            bits = 0;
            wubu_encode_frame(frames + f*np, ref, NULL, W, H, qp, WUBU_P_FRAME, NULL, recon, &bits);
            total_bits += bits;
            memcpy(ref, recon, np);
        }
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double fps = NF / elapsed;
        printf("%-20s QP=%d: %6.1f fps, %7.1f kbps, %ld bits/frame (incl I)\n",
               preset_names[1], qp, fps, total_bits * 30.0 / NF / 1024.0, total_bits/NF);
        free(recon);
        free(ref);
    }
    
    /* B-frame GOP-8 benchmark */
    {
        int qp = 26;
        uint8_t* recon = malloc((size_t)9 * np);
        clock_t start = clock();
        long total_bits = 0;
        wubu_encode_gop8(frames, W, H, qp, recon, &total_bits);
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double fps = 9 / elapsed;
        printf("%-20s QP=%d: %6.1f fps, %7.1f kbps, %ld bits/frame (GOP-8)\n",
               preset_names[2], qp, fps, total_bits * 30.0 / 9 / 1024.0, total_bits/9);
        free(recon);
    }
    
    /* QP sweep for P-frames */
    printf("\n=== P-frame RD curve (QP sweep) ===\n");
    for(int q=0; q<4; q++){
        int qp = qps[q];
        uint8_t* recon = malloc(np);
        uint8_t* ref = malloc(np);
        long total_bits = 0;
        long total_mse = 0;
        long bits;
        wubu_encode_frame(frames, NULL, NULL, W, H, qp, WUBU_I_FRAME, NULL, ref, &bits);
        total_bits = bits;
        for(int f=1; f<NF; f++){
            bits = 0;
            wubu_encode_frame(frames + f*np, ref, NULL, W, H, qp, WUBU_P_FRAME, NULL, recon, &bits);
            total_bits += bits;
            for(int p=0; p<np; p++){
                int d = frames[f*np+p] - recon[p];
                total_mse += d*d;
            }
            memcpy(ref, recon, np);
        }
        double avg_psnr = total_mse > 0 ? 10.0 * log10(255.0*255.0 * (NF-1) * np / (double)total_mse) : 99.0;
        printf("QP=%2d: %7ld bits/frame, %5.1f dB\n", qp, total_bits/NF, avg_psnr);
        free(recon);
        free(ref);
    }
    
    free(frames);
    return 0;
}

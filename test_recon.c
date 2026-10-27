/* test_recon.c — Test if --recon flag works */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

double compute_psnr(const uint8_t* a, const uint8_t* b, size_t n){
    double mse = 0;
    for(size_t i=0; i<n; i++){ double d = (double)a[i] - (double)b[i]; mse += d*d; }
    mse /= n;
    return mse > 0 ? 10.0 * log10(255.0*255.0/mse) : 99.0;
}

int main(){
    int W = 320, H = 240, n = 360;
    int fs = W*H;
    
    /* Read original */
    FILE* f = fopen("/tmp/dbz_ab/dbz_input.raw", "rb");
    if(!f){ printf("Cannot open input\n"); return 1; }
    uint8_t* orig = malloc((size_t)n * fs);
    fread(orig, 1, (size_t)n * fs, f);
    fclose(f);
    
    /* Read recon */
    f = fopen("/tmp/dbz_ab/test5_recon.raw", "rb");
    if(!f){ printf("Cannot open recon file\n"); return 1; }
    uint8_t* recon = malloc((size_t)n * fs);
    fread(recon, 1, (size_t)n * fs, f);
    fclose(f);
    
    /* Compute PSNR */
    double psnr = compute_psnr(orig, recon, (size_t)n * fs);
    printf("PSNR: %.2f dB\n", psnr);
    
    /* Check first frame */
    double psnr0 = compute_psnr(orig, recon, fs);
    printf("Frame 0 PSNR: %.2f dB\n", psnr0);
    
    free(orig);
    free(recon);
    return 0;
}

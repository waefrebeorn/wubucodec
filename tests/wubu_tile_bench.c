/* wubu_tile_bench.c — benchmark tile-based parallelism scaling */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "wubu_encoder.h"
#include "wubu_tile_parallel.h"

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

#define W 1920
#define H 1080
#define NF 12

int main(void){
    const int np = W*H;
    uint8_t* frames = malloc((size_t)NF * np);
    
    for(int f=0; f<NF; f++){
        for(int y=0; y<H; y++){
            for(int x=0; x<W; x++){
                float edge = 960.0f + f * 0.7f;
                float v = (x < edge) ? 100.0f : 200.0f;
                v += 20.0f * sinf(2.0f * M_PI * (x + f) / W) * cosf(2.0f * M_PI * y / H);
                if(v<0)v=0; if(v>255)v=255;
                frames[(size_t)f*np + y*W + x] = (uint8_t)v;
            }
        }
    }
    
    printf("=== WuBuCodec Tile-Parallel Benchmark (1920x1080) ===\n\n");
    
    int tile_counts[] = {1, 2, 4, 8};
    for(int t=0; t<4; t++){
        int num_tiles = tile_counts[t];
        uint8_t* recon = malloc(np);
        clock_t start = clock();
        long total_bits = 0;
        
        for(int f=0; f<NF; f++){
            long bits;
            wubu_encode_frame_tiled(frames + f*np, NULL, W, H, 26, recon, &bits, num_tiles, 1);
            total_bits += bits;
        }
        
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
        double fps = NF / elapsed;
        
        printf("%d tile(s): %6.1f fps, %7.1f kbps\n", num_tiles, fps, total_bits * 30.0 / NF / 1024.0);
        free(recon);
    }
    
    free(frames);
    return 0;
}

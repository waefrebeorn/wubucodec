/* gen_test_content.c — Generate test content for codec evaluation
 *
 * Generates 4 types of test content:
 * 1. gradient: smooth gradient (tests basic DCT)
 * 2. anime: sharp edges + flat regions (tests intra prediction)
 * 3. screen: text + UI elements (tests variable block sizes)
 * 4. movie: natural video with grain (tests inter prediction + grain)
 *
 * Usage: ./gen_test_content <output_dir> <W> <H> <frames>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void gen_gradient(uint8_t *f, int W, int H, int seed, float t) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float v = 128.0f + 64.0f * sinf(2.0f * M_PI * (float)x / W + t) *
                                      cosf(2.0f * M_PI * (float)y / H + t);
            f[y * W + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
}

static void gen_anime(uint8_t *f, int W, int H, int seed, float t) {
    /* Sharp edges + flat regions */
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int v;
            float phase = t * 0.5f;
            if ((x + (int)(phase * 20)) % 64 < 32 && (y + (int)(phase * 10)) % 48 < 24) {
                v = 200; /* bright region */
            } else if ((x + (int)(phase * 20)) % 64 < 32 || (y + (int)(phase * 10)) % 48 < 24) {
                v = 50; /* dark region */
            } else {
                v = 128; /* mid region */
            }
            /* Add diagonal edge */
            if (abs(x - y + (int)(phase * 30)) < 3) v = 255;
            if (abs(x + y - W + (int)(phase * 20)) < 3) v = 0;
            f[y * W + x] = (uint8_t)v;
        }
}

static void gen_screen(uint8_t *f, int W, int H, int seed, float t) {
    /* Text-like patterns */
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int v = 240; /* white background */
            /* Horizontal lines (text rows) */
            if (y % 16 < 2 && x % 4 < 2) v = 30;
            /* Vertical lines (characters) */
            if (x % 8 < 2 && y % 16 < 12) v = 30;
            /* Moving cursor */
            if (abs(y - H/2) < 2 && abs(x - (int)(t * 50) % W) < 4) v = 0;
            f[y * W + x] = (uint8_t)v;
        }
}

static void gen_movie(uint8_t *f, int W, int H, int seed, float t) {
    /* Natural-like content with motion */
    srand(seed);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float v = 128.0f + 32.0f * sinf(2.0f * M_PI * (float)x / W * 2 + t) *
                                      cosf(2.0f * M_PI * (float)y / H * 3 + t * 0.7f);
            /* Add some noise (film grain) */
            v += (float)((rand() % 40) - 20);
            f[y * W + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s <output_dir> <W> <H> <frames>\n", argv[0]);
        return 1;
    }
    
    const char *dir = argv[1];
    int W = atoi(argv[2]);
    int H = atoi(argv[3]);
    int nf = atoi(argv[4]);
    int np = W * H;
    
    uint8_t *f = (uint8_t *)malloc(np);
    
    for (int type = 0; type < 4; type++) {
        const char *names[] = { "gradient", "anime", "screen", "movie" };
        char path[256];
        snprintf(path, sizeof(path), "%s/%s_%dx%d_%dfr.raw", dir, names[type], W, H, nf);
        
        FILE *out = fopen(path, "wb");
        if (!out) { printf("Cannot create %s\n", path); continue; }
        
        for (int fn = 0; fn < nf; fn++) {
            float t = (float)fn / (float)nf;
            switch (type) {
                case 0: gen_gradient(f, W, H, 42, t); break;
                case 1: gen_anime(f, W, H, 42, t); break;
                case 2: gen_screen(f, W, H, 42, t); break;
                case 3: gen_movie(f, W, H, 42, t); break;
            }
            fwrite(f, 1, np, out);
        }
        fclose(out);
        printf("Generated %s (%d frames)\n", path, nf);
    }
    
    free(f);
    return 0;
}

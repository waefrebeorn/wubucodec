/* test_av2.c: Minimal A/V test - copy working pattern from gap_closure */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_encoder.h"
#include "wubuv.h"

#define W 320
#define H 240
#define FPS 30
#define N 10

int main(void) {
    printf("=== Minimal A/V Test ===\n");
    
    const int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    
    /* Generate simple frame */
    for(int i = 0; i < np; i++) {
        int x = i % W;
        int y = i / W;
        frame[i] = (uint8_t)((x + y) % 256);
    }
    
    /* Test direct encoder (like gap_closure does) */
    printf("Testing wubu_encode_frame...\n");
    long bits = 0;
    int rc = wubu_encode_frame(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits);
    printf("  rc=%d, bits=%ld\n", rc, bits);
    
    /* Test WUBV writer */
    printf("Testing WUBV writer...\n");
    WubuvHeader hdr;
    wubuv_hdr_init(&hdr, W, H, FPS, N, 0, 0);
    
    WubuvWriter* writer = wubuv_writer_open("docs/video_examples/mini.wubv", &hdr);
    if(!writer) {
        printf("  Failed to create writer\n");
        return 1;
    }
    
    rc = wubuv_write_frame(writer, recon, 0);
    printf("  Write frame: rc=%d\n", rc);
    
    wubuv_writer_close(writer);
    printf("  Writer closed\n");
    
    /* Test WUBV reader */
    printf("Testing WUBV reader...\n");
    WubuvReader* reader = wubuv_reader_open("docs/video_examples/mini.wubv");
    if(!reader) {
        printf("  Failed to open reader\n");
        return 1;
    }
    
    uint8_t* read_frame = malloc(np);
    int is_inter;
    rc = wubuv_read_frame(reader, read_frame, &is_inter);
    printf("  Read frame: rc=%d, is_inter=%d\n", rc, is_inter);
    
    wubuv_reader_close(reader);
    free(read_frame);
    
    free(frame);
    free(recon);
    
    printf("=== Done ===\n");
    return 0;
}
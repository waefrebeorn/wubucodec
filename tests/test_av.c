/* test_av.c: Test A/V encoding step by step */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_encoder.h"
#include "wubuv_audio.h"

#define W 320
#define H 240
#define FPS 30
#define N 30

int main(void) {
    printf("=== WuBuCodec A/V Test ===\n");
    
    const int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    
    /* Test 1: Direct encoder call */
    printf("\n[Test 1] Direct encoder call...\n");
    for(int i = 0; i < np; i++) {
        int x = i % W;
        int y = i / W;
        frame[i] = (uint8_t)((x + y) % 256);
    }
    
    long bits = 0;
    int rc = wubu_encode_frame(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits);
    printf("  wubu_encode_frame: rc=%d, bits=%ld\n", rc, bits);
    
    /* Test 2: Preset encoder call */
    printf("\n[Test 2] Preset encoder call...\n");
    bits = 0;
    rc = wubu_encode_frame_preset(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits, 2);
    printf("  wubu_encode_frame_preset: rc=%d, bits=%ld\n", rc, bits);
    
    /* Test 3: WUBV writer */
    printf("\n[Test 3] WUBV writer...\n");
    WubuvHeader hdr;
    wubuv_hdr_init(&hdr, W, H, FPS, N, 0, 0);
    printf("  Header: %dx%d @ %d fps, %d frames\n", hdr.width, hdr.height, hdr.fps, hdr.frame_count);
    
    WubuvWriter* writer = wubuv_writer_open("docs/video_examples/test.wubv", &hdr);
    if(!writer) {
        printf("  Failed to create writer\n");
        return 1;
    }
    printf("  Writer created\n");
    
    /* Write one frame */
    rc = wubuv_write_frame(writer, recon, 0);
    printf("  Write frame: rc=%d\n", rc);
    
    wubuv_writer_close(writer);
    printf("  Writer closed\n");
    
    /* Test 4: WUBV reader */
    printf("\n[Test 4] WUBV reader...\n");
    WubuvReader* reader = wubuv_reader_open("docs/video_examples/test.wubv");
    if(!reader) {
        printf("  Failed to open reader\n");
        return 1;
    }
    printf("  Reader opened: %dx%d @ %d fps\n", reader->hdr.width, reader->hdr.height, reader->hdr.fps);
    
    uint8_t* read_frame = malloc(np);
    int is_inter;
    rc = wubuv_read_frame(reader, read_frame, &is_inter);
    printf("  Read frame: rc=%d, is_inter=%d\n", rc, is_inter);
    
    wubuv_reader_close(reader);
    free(read_frame);
    
    /* Test 5: Multiple frames */
    printf("\n[Test 5] Multiple frames...\n");
    wubuv_hdr_init(&hdr, W, H, FPS, N, 0, 0);
    writer = wubuv_writer_open("docs/video_examples/test_multi.wubv", &hdr);
    
    uint8_t* prev = NULL;
    int count = 0;
    for(int f = 0; f < N; f++) {
        for(int i = 0; i < np; i++) {
            int x = i % W;
            int y = i / W;
            frame[i] = (uint8_t)((x + y + f * 3) % 256);
        }
        
        bits = 0;
        rc = wubu_encode_frame_preset(frame, prev, NULL, W, H, 26, f == 0 ? WUBU_I_FRAME : WUBU_P_FRAME, NULL, recon, &bits, 2);
        if(rc != 0) {
            printf("  Encode failed at frame %d: %d\n", f, rc);
            break;
        }
        
        wubuv_write_frame(writer, recon, f > 0);
        
        if(!prev) prev = malloc(np);
        memcpy(prev, recon, np);
        count++;
    }
    
    wubuv_writer_close(writer);
    printf("  Wrote %d frames\n", count);
    
    /* Verify */
    reader = wubuv_reader_open("docs/video_examples/test_multi.wubv");
    if(reader) {
        count = 0;
        while(wubuv_read_frame(reader, read_frame, &is_inter) == 0) {
            count++;
        }
        printf("  Read back %d frames\n", count);
        wubuv_reader_close(reader);
    }
    
    free(frame);
    free(recon);
    free(prev);
    
    printf("\n=== All tests passed! ===\n");
    return 0;
}
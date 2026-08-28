/* av_demo.c: Simple A/V encoding demo */
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
    printf("=== WuBuCodec A/V Demo ===\n");
    
    const int np = W * H;
    uint8_t* frame = malloc(np);
    uint8_t* recon = malloc(np);
    
    /* Test encoder directly first */
    printf("Testing encoder...\n");
    for(int i = 0; i < np; i++) {
        int x = i % W;
        int y = i / W;
        frame[i] = (uint8_t)((x + y) % 256);
    }
    
    long bits = 0;
    int rc = wubu_encode_frame(frame, NULL, NULL, W, H, 26, WUBU_I_FRAME, NULL, recon, &bits);
    printf("  Encode result: %d, bits: %ld\n", rc, bits);
    
    /* Now test WUBV writer */
    printf("Testing WUBV writer...\n");
    WubuvHeader hdr;
    wubuv_hdr_init(&hdr, W, H, FPS, N, 0, 0);
    printf("  Header initialized\n");
    
    WubuvWriter* writer = wubuv_writer_open("docs/video_examples/demo.wubv", &hdr);
    if(!writer) {
        printf("Failed to create writer\n");
        return 1;
    }
    printf("  Writer created\n");
    
    /* Write frames */
    for(int f = 0; f < N; f++) {
        for(int i = 0; i < np; i++) {
            int x = i % W;
            int y = i / W;
            frame[i] = (uint8_t)((x + y + f * 3) % 256);
        }
        
        long bits = 0;
        int frame_type = (f == 0) ? WUBU_I_FRAME : WUBU_P_FRAME;
        uint8_t* ref = (f > 0) ? recon : NULL;
        
        rc = wubu_encode_frame(frame, ref, NULL, W, H, 26, frame_type, NULL, recon, &bits);
        if(rc != 0) {
            printf("Encode failed at frame %d: %d\n", f, rc);
            break;
        }
        
        wubuv_write_frame(writer, recon, f > 0);
    }
    
    wubuv_writer_close(writer);
    printf("Created demo.wubv\n");
    
    /* Also save raw video for ffmpeg */
    printf("Saving raw video...\n");
    FILE* fraw = fopen("docs/video_examples/original_video.raw", "wb");
    FILE* fenc = fopen("docs/video_examples/encoded_qp26.raw", "wb");
    for(int f = 0; f < N; f++) {
        for(int i = 0; i < np; i++) {
            int x = i % W;
            int y = i / W;
            frame[i] = (uint8_t)((x + y + f * 3) % 256);
        }
        fwrite(frame, 1, np, fraw);
        
        long bits = 0;
        int frame_type = (f == 0) ? WUBU_I_FRAME : WUBU_P_FRAME;
        uint8_t* ref = (f > 0) ? recon : NULL;
        wubu_encode_frame(frame, ref, NULL, W, H, 26, frame_type, NULL, recon, &bits);
        fwrite(recon, 1, np, fenc);
    }
    fclose(fraw);
    fclose(fenc);
    
    /* Read back and verify */
    printf("Verifying...\n");
    WubuvReader* reader = wubuv_reader_open("docs/video_examples/demo.wubv");
    if(reader) {
        printf("  WUBV: %dx%d @ %d fps, %d frames\n", 
               reader->hdr.width, reader->hdr.height, reader->hdr.fps, reader->hdr.frame_count);
        
        uint8_t* read_frame = malloc(np);
        int count = 0;
        int is_inter;
        while(wubuv_read_frame(reader, read_frame, &is_inter) == 0) {
            count++;
        }
        printf("  Read back %d frames\n", count);
        free(read_frame);
        wubuv_reader_close(reader);
    }
    
    free(frame);
    free(recon);
    
    printf("Done! Convert to MP4 with:\n");
    printf("  ffmpeg -f rawvideo -pix_fmt gray -s 320x240 -r 30 -i original_video.raw -c:v libx264 original_video.mp4\n");
    
    return 0;
}
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include "wubuv.h"

int main() {
    printf("sizeof(WubuvHeader) = %zu\n", sizeof(WubuvHeader));
    
    WubuvHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "WUBV", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.width = 320;
    hdr.height = 240;
    hdr.fps = 30;
    hdr.frame_count = 10;
    hdr.audio_rate = 0;
    
    printf("magic: %c%c%c%c\n", hdr.magic[0], hdr.magic[1], hdr.magic[2], hdr.magic[3]);
    printf("version: %d\n", hdr.version);
    printf("width: %d, height: %d\n", hdr.width, hdr.height);
    printf("fps: %d, frames: %d\n", hdr.fps, hdr.frame_count);
    
    // Write test
    FILE* f = fopen("docs/video_examples/debug.wubv", "wb");
    if(!f) { printf("Cannot open\n"); return 1; }
    fwrite(&hdr, sizeof(hdr), 1, f);
    printf("Written %zu bytes\n", sizeof(hdr));
    fclose(f);
    
    // Read back
    f = fopen("docs/video_examples/debug.wubv", "rb");
    if(!f) { printf("Cannot open\n"); return 1; }
    WubuvHeader hdr2;
    memset(&hdr2, 0, sizeof(hdr2));
    fread(&hdr2, sizeof(hdr2), 1, f);
    fclose(f);
    
    printf("Read back:\n");
    printf("magic: %c%c%c%c\n", hdr2.magic[0], hdr2.magic[1], hdr2.magic[2], hdr2.magic[3]);
    printf("version: %d\n", hdr2.version);
    printf("width: %d, height: %d\n", hdr2.width, hdr2.height);
    
    // Test validation
    printf("valid: %d\n", wubuv_hdr_valid(&hdr2));
    
    return 0;
}

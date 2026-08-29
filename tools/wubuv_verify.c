/* wubuv_verify.c — Verify WUBV container integrity */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include "wubuv.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s file.wubv\n", argv[0]);
        return 1;
    }
    
    printf("Verifying %s...\n", argv[1]);
    
    if (wubuv_verify(argv[1])) {
        printf("  CRC: VALID\n");
        
        /* Also try to read all frames */
        WubuvReader* r = wubuv_reader_open(argv[1]);
        if (r) {
            uint8_t type;
            uint8_t* data;
            uint32_t size;
            int video_frames = 0, audio_frames = 0, errors = 0;
            
            while (wubuv_read_frame(r, &type, &data, &size) == 0) {
                if (type == 0 || type == 1) video_frames++;
                else if (type == 2) audio_frames++;
                free(data);
            }
            
            printf("  Video frames: %d\n", video_frames);
            printf("  Audio frames: %d\n", audio_frames);
            printf("  Result: OK\n");
            
            wubuv_reader_close(r);
        }
        
        return 0;
    } else {
        printf("  CRC: INVALID\n");
        return 1;
    }
}

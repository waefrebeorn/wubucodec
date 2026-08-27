/* test_wubuv.c -- prove the .WUBV format works
 *  G1 header round trip: write, read, magic+fields match
 *  G2 frame round trip: KEY then INTER frames decode byte-exact
 *  G3 CRC integrity: untouched file verifies, corrupted bit fails
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubuv.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== .WUBV Format Tests ===\n\n");
    const char* PATH="/tmp/wubuv_test.wubv";
    const int W=64,H=48,FPS=30,NFRAMES=5;

    /* synthetic video: moving gradient + noise */
    uint8_t frames[NFRAMES][W*H*3];
    unsigned rs=42u;
    for(int f2=0;f2<NFRAMES;f2++)
        for(int i=0;i<W*H*3;i++){
            rs=rs*1103515245u+12345u;
            frames[f2][i]=(uint8_t)(((i+f2*7)%255)+((rs>>16)%8)-4);
        }

    printf("  g1_header_roundtrip...");
    {
        WubuvHeader h;
        wubuv_hdr_init(&h,W,H,FPS,NFRAMES,0,0);
        WubuvWriter* w=wubuv_writer_open(PATH,&h);
        CHECK(w!=NULL);
        for(int f2=0;f2<NFRAMES;f2++)
            wubuv_write_frame(w,frames[f2],f2>0);   /* first = KEY */
        wubuv_writer_close(w);

        WubuvReader* r=wubuv_reader_open(PATH);
        CHECK(r!=NULL);
        CHECK(memcmp(r->hdr.magic,"WUBV",4)==0);
        CHECK(r->hdr.width==W&&r->hdr.height==H);
        CHECK(r->hdr.fps==FPS&&r->hdr.frame_count==NFRAMES);
        wubuv_reader_close(r);
    }
    printf("PASS\n");passed++;

    printf("  g2_frame_roundtrip_exact...");
    {
        WubuvReader* r=wubuv_reader_open(PATH);
        CHECK(r!=NULL);
        uint8_t buf[W*H*3];
        for(int f2=0;f2<NFRAMES;f2++){
            int inter;
            CHECK(wubuv_read_frame(r,buf,&inter)==0);
            /* byte-exact reconstruction */
            for(size_t i=0;i<(size_t)W*H*3;i++)
                if(buf[i]!=frames[f2][i]){
                    printf("[byte %zu: got %u want %u] ",
                           (size_t)i,buf[i],frames[f2][i]);
                    failed++;
                    return 1;
                }
        }
        wubuv_reader_close(r);
    }
    printf("PASS\n");passed++;

    printf("  g3_crc_integrity...");
    {
        /* pristine file verifies */
        CHECK(wubuv_verify(PATH)==1);
        /* corrupt one byte in the middle → must fail */
        FILE* f=fopen(PATH,"r+b");
        fseek(f,300,SEEK_SET);
        int orig=fgetc(f);
        fseek(f,300,SEEK_SET);
        fputc(orig^0xFF,f);
        fclose(f);
        CHECK(wubuv_verify(PATH)==0);
        /* restore for cleanliness */
        f=fopen(PATH,"r+b");
        fseek(f,300,SEEK_SET);
        fputc(orig,f);
        fclose(f);
    }
    printf("PASS\n");passed++;

    /* size report */
    {
        FILE* f=fopen(PATH,"rb");
        fseek(f,0,SEEK_END);
        long sz=ftell(f);
        fclose(f);
        float raw=(float)(NFRAMES*W*H*3);
        printf("\n  [raw=%.0f bytes, file=%ld bytes, ratio=%.1fx]\n",
               (double)raw,sz,(double)(raw/sz));
    }

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

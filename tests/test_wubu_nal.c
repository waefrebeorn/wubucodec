#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_nal.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== NAL + MP4 Tests ===\n\n");

    printf("  g1_parse_annexb...");
    {
        /* construct a simple Annex B stream: SPS + PPS + IDR */
        uint8_t stream[]={
            0x00,0x00,0x01, 0x67, 1,2,3,       /* SPS */
            0x00,0x00,0x01, 0x68, 4,5,          /* PPS */
            0x00,0x00,0x00,0x01, 0x65, 6,7,8,9 /* IDR with 4-byte SC */
        };
        long stream_size=sizeof(stream);
        
        WubuNalUnit units[8];
        int n=wubu_nal_parse_annexb(stream,stream_size,units,8);
        printf("[%d NAL units] ",n);
        CHECK(n==3);
        CHECK(units[0].type==7);  /* SPS */
        CHECK(units[1].type==8);  /* PPS */
        CHECK(units[2].type==5);           /* IDR */
    }
    printf("PASS\n");passed++;

    printf("  g2_emulation_prevention...");
    {
        uint8_t raw[]={0x00,0x00,0x00,0x42}; /* contains start code pattern */
        uint8_t escaped[16];
        uint8_t unescaped[16];
        
        long esc_len=wubu_nal_add_ep(raw,escaped,4);
        long raw_len=wubu_nal_remove_ep(escaped,unescaped,esc_len);
        
        CHECK(raw_len==4);
        CHECK(memcmp(raw,unescaped,4)==0); /* round trip */
    }
    printf("PASS\n");passed++;

    printf("  g3_ftyp_box...");
    {
        uint8_t buf[32];
        long size=wubu_mp4_ftyp(buf,sizeof(buf));
        CHECK(size>0&&size<=20);
        /* check box type at offset 4 */
        CHECK(memcmp(buf+4,"ftyp",4)==0);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

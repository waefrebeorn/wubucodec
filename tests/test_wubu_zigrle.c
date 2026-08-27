#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_zigrle.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Zigzag + RLE Tests ===\n\n");

    printf("  g1_zigzag_roundtrip...");
    {
        int block[64],scanned[64],restored[64];
        for(int i=0;i<64;i++)block[i]=i+1;
        wubu_zz_scan(block,scanned);
        wubu_zz_unscan(scanned,restored);
        for(int i=0;i<64;i++)CHECK(restored[i]==block[i]);
    }
    printf("PASS\n");passed++;

    printf("  g2_rle_smooth_block...");
    {
        /* smooth block: most high-freq are zero after DCT+quantize */
        int scanned[64]={5,-2,1,0,0,0,1,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          -1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3};
        int runs[64],values[64];
        int n=wubu_rle_encode(scanned,64,runs,values);
        printf("[%d pairs from 64 coeffs] ",n);
        CHECK(n<10);  /* should compress well */
        /* decode back */
        int decoded[64];
        wubu_rle_decode(runs,values,n,decoded,64);
        for(int i=0;i<64;i++)CHECK(decoded[i]==scanned[i]);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

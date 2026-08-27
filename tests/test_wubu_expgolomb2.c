#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_expgolomb2.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Exp-Golomb v2 Tests ===\n\n");

    printf("  g1_roundtrip...");
    {
        int values[]={0,1,-1,2,-2,3,-3,10,-10,100,-100,0,0,0};
        int n=14;
        uint8_t* buf;
        long size=wubu_eg2_encode_array(values,n,&buf);
        CHECK(size>0);
        int decoded[64];
        int nd=wubu_eg2_decode_array(buf,size,decoded,64);
        CHECK(nd>=n);
        for(int i=0;i<n;i++)CHECK(decoded[i]==values[i]);
        printf("[%ld bytes for %d values] ",size,n);
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("  g2_small_values_short...");
    {
        /* zeros and small values should produce very few bits */
        int values[]={0,0,0,0,1,-1,0,0};
        int n=8;
        uint8_t* buf;
        long size=wubu_eg2_encode_array(values,n,&buf);
        /* 8 values × ~3 bits average = ~24 bits = 3 bytes */
        printf("[%d values in %ld bits] ",n,size*8);
        CHECK(size<=4);
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

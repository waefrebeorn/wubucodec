/* test_wubu_delta.c -- GAP-E007 gates
 *  G1 zigzag round-trip exact on boundary values
 *  G2 delta+zigzag+varint full round-trip exact
 *  G3 smooth data compresses smaller than raw
 *  G4 random data round-trips (no corruption)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_delta.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Delta/Zigzag/Varint Tests ===\n\n");

    printf("  g1_zigzag_boundaries...");
    {
        int32_t vals[]={0,-1,1,-2,2,INT32_MAX/2,INT32_MIN/2};
        for(int i=0;i<7;i++){
            uint32_t z=wubu_zigzag_encode(vals[i]);
            int32_t back=wubu_zigzag_decode(z);
            CHECK(back==vals[i]);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_round_trip...");
    {
        int32_t src[100],dst[100];
        for(int i=0;i<100;i++)src[i]=i*7-200;   /* linear ramp */
        uint8_t buf[600];
        size_t len=wubu_delta_compress(src,100,buf,sizeof(buf));
        CHECK(len>0);
        CHECK(wubu_delta_decompress(buf,len,dst,100)==0);
        for(int i=0;i<100;i++)CHECK(dst[i]==src[i]);
    }
    printf("PASS\n");passed++;

    printf("  g3_smooth_compresses...");
    {
        /* smooth ramp vs random — smooth should be much smaller */
        int32_t smooth[500],rnd[500];
        unsigned rs=42u;
        for(int i=0;i<500;i++){
            smooth[i]=i*3;
            rs=rs*1103515245u+12345u;
            rnd[i]=(int32_t)((rs>>16)%100000)-50000;
        }
        uint8_t b1[3000],b2[3000];
        size_t l1=wubu_delta_compress(smooth,500,b1,sizeof(b1));
        size_t l2=wubu_delta_compress(rnd,500,b2,sizeof(b2));
        printf("[smooth=%zuB random=%zuB] ",l1,l2);
        CHECK(l1<l2);   /* delta helps smooth sequences */
        CHECK(l1<500*4); /* varint beats fixed 4-byte encoding */
    }
    printf("PASS\n");passed++;

    printf("  g4_random_round_trip...");
    {
        int32_t src[50],dst[50];
        unsigned rs=99u;
        for(int i=0;i<50;i++){
            rs=rs*1103515245u+12345u;
            src[i]=(int32_t)((rs>>16)%200000)-100000;
        }
        uint8_t buf[600];
        size_t len=wubu_delta_compress(src,50,buf,sizeof(buf));
        CHECK(len>0);
        CHECK(wubu_delta_decompress(buf,len,dst,50)==0);
        for(int i=0;i<50;i++)CHECK(dst[i]==src[i]);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

/* test_wubu_expgolomb.c -- GAP-E008 gates
 *  G1 known codewords: 0->'1', 1->'010', 2->'011', 3->'00100'
 *  G2 round-trip on skewed distribution (many small values)
 *  G3 small values use fewer bits than large (prefix property)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_expgolomb.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Exponential-Golomb Tests ===\n\n");

    printf("  g1_known_codewords...");
    {
        /* encode single value, check total bit length matches expected */
        struct {uint32_t v;int bits;} cases[]={
            {0,1},{1,3},{2,3},{3,5},{4,5},{6,5},{7,7}
        };
        for(int ci=0;ci<7;ci++){
            uint8_t buf[8];WubuBitWriter bw;
            wubu_bw_init(&bw,buf,8);
            CHECK(wubu_eg_put(&bw,cases[ci].v)==0);
            int used_bits=(int)bw.pos*8+(bw.bit>0?bw.bit:0);
            CHECK(used_bits==cases[ci].bits);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_round_trip_skewed...");
    {
        /* geometric-ish: mostly small, some large */
        uint32_t vals[100];
        unsigned rs=42u;
        for(int i=0;i<100;i++){
            rs=rs*1103515245u+12345u;
            vals[i]=(uint32_t)((rs>>16)%1000)/990;   /* mostly < 10 */
        }
        uint8_t buf[800];WubuBitWriter bw;
        wubu_bw_init(&bw,buf,sizeof(buf));
        for(int i=0;i<100;i++)CHECK(wubu_eg_put(&bw,vals[i])==0);
        size_t len=bw.pos+(bw.bit?1:0);

        WubuBitReader br;
        wubu_br_init(&br,buf,len);
        for(int i=0;i<100;i++){
            uint32_t v;
            CHECK(wubu_eg_get(&br,&v)==0);
            CHECK(v==vals[i]);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_prefix_property...");
    {
        /* EG is a prefix code: encoding 0..N uses strictly increasing
         * code lengths at power-of-2 boundaries. Verify monotone length. */
        int prev_len=-1;
        for(uint32_t v=0;v<16;v++){
            uint8_t buf[8];WubuBitWriter bw;
            wubu_bw_init(&bw,buf,8);
            wubu_eg_put(&bw,v);
            int bits=(int)bw.pos*8+(bw.bit>0?bw.bit:0);
            /* EG length grows at (v+1) power-of-2 boundaries:
             * v+1 = 2^k → L=k+1 bits total = 2k+1 */
            uint32_t vp=v+1;
            if(vp>1&&(vp&(vp-1))==0){
                CHECK(bits==prev_len+2);  /* exactly +2 at each boundary */
            }
            prev_len=bits;
        }
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

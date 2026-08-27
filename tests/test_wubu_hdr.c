#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_hdr.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== HDR Tests ===\n\n");

    printf("  g1_pq_known_values...");
    {
        /* PQ value 400 at 10-bit ≈ 29.4 nits (from published table) */
        double nits=wubu_pq_eotf_full(400,10);
        printf("[pq400=%.1f nits (want ~29.4)] ",nits);
        CHECK(nits>25&&nits<35);
        
        /* PQ value 769 at 10-bit ≈ 1000 nits */
        double nits_1k=wubu_pq_eotf_full(769,10);
        printf("pq769=%.1f (want ~999) ",nits_1k);
        CHECK(nits_1k>900&&nits_1k<1100);
    }
    printf("PASS\n");passed++;

    printf("  g2_pq_roundtrip...");
    {
        for(int v=50;v<1024;v+=200){
            double nits=wubu_pq_eotf_full(v,10);
            int back=wubu_pq_inverse_eotf(nits,10);
            if(abs(back-v)>2){printf("[v=%d back=%d FAIL] ",v,back);CHECK(0);}
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_hlg_monotone...");
    {
        /* HLG output should increase monotonically with input */
        double prev=0;
        for(int i=1;i<=20;i++){
            double x=(double)i/12.0;
            double out=wubu_hlg_oetf(x);
            if(out<=prev){printf("[x=%.3f out=%.4f <= prev %.4f] ",x,out,prev);CHECK(0);}
            prev=out;
        }
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

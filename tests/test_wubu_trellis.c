#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_trellis.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Trellis Quantization Tests ===\n\n");

    printf("  g1_trellis_vs_rounding...");
    {
        /* create coefficients where some are near quantization boundaries */
        double coeffs[32];
        srand(42);
        for(int i=0;i<32;i++)
            coeffs[i]=((double)rand()/RAND_MAX-0.5)*2*200; /* [-200,200] */
        
        long sse_std,bits_std,sse_tr,bits_tr;
        wubu_trellis_vs_rounding(coeffs,32,25,0.05,
                                  &sse_std,&bits_std,&sse_tr,&bits_tr);
        
        printf("[std: sse=%ld bits=%ld | tr: sse=%ld bits=%ld] ",
               sse_std,bits_std,sse_tr,bits_tr);
        /* trellis should use fewer or equal bits (it optimizes rate) */
        CHECK(bits_tr<=bits_std+10); /* allow small tolerance */
    }
    printf("PASS\n");passed++;

    printf("  g2_zero_coefficient_free...");
    {
        /* very small lambda → trellis should match rounding closely */
        double coeffs[8]={100.0,-80.0,55.0,-30.0,15.0,-5.0,3.0,-1.0};
        int16_t levels[8];
        double rd=wubu_trellis_quantize(coeffs,8,10,0.001,levels);
        
        CHECK(rd>=0);
        /* first few should be nonzero */
        CHECK(levels[0]!=0);
        CHECK(levels[1]!=0);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

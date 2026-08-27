#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_transform.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Variable-size Transform Tests ===\n\n");

    printf("  g1_roundtrip_4x4...");
    {
        int16_t input[16],coeff[16],output[16];
        for(int i=0;i<16;i++)input[i]=(int16_t)((i*37)%256-128);
        
        wubu_tr_forward(input,coeff,4);
        wubu_tr_inverse(coeff,output,4);
        
        long err=0;
        for(int i=0;i<16;i++){
            long d=labs(output[i]-input[i]);
            if(d>err)err=d;
        }
        printf("[max_err=%ld] ",err);
        CHECK(err<4); /* rounding error only */
    }
    printf("PASS\n");passed++;

    printf("  g2_roundtrip_16x16...");
    {
        int n=16;
        int16_t input[256],coeff[256],output[256];
        for(int i=0;i<n*n;i++)input[i]=(int16_t)((i*13)%512-256);
        
        wubu_tr_forward(input,coeff,n);
        wubu_tr_inverse(coeff,output,n);
        
        long max_err=0;
        for(int i=0;i<n*n;i++){
            long d=labs(output[i]-input[i]);
            if(d>max_err)max_err=d;
        }
        printf("[max_err=%ld] ",max_err);
        CHECK(max_err<16); /* larger block = slightly more rounding */
    }
    printf("PASS\n");passed++;

    printf("  g3_dc_concentrates_energy...");
    {
        /* constant input → only DC coefficient should be non-zero */
        int16_t input[64]={0},coeff[64]={0};
        for(int i=0;i<64;i++)input[i]=100;
        
        wubu_tr_forward(input,coeff,8);
        
        /* DC should be 8*100=800, AC should be ~0 */
        CHECK(labs(coeff[0])>700); /* DC is large */
        int nonzero_ac=0;
        for(int i=1;i<64;i++)
            if(abs(coeff[i])>10)nonzero_ac++;
        CHECK(nonzero_ac==0); /* all AC near zero */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

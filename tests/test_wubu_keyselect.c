#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_keyselect.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Keyframe Selection Tests ===\n\n");
    const int NF=20,D=4;

    /* rotation sequence with a FAST section in the middle */
    float quats[NF*D];
    for(int i=0;i<NF;i++){
        float rate=(i>=8&&i<=12)?0.15f:0.03f;  /* fast in the middle */
        float half=i*rate/2;
        quats[i*D+0]=cosf(half);
        quats[i*D+1]=0;quats[i*D+2]=0;
        quats[i*D+3]=sinf(half);
    }

    printf("  g1_selects_fast_section...");
    {
        int keys[5];
        int n=wubu_kf_select(quats,NF,D,5,keys);
        CHECK(n==5);
        /* keys should cluster around frames 8-12 (the fast part) */
        int in_fast=0;
        for(int i=0;i<n;i++)
            if(keys[i]>=7&&keys[i]<=13)in_fast++;
        printf("[%d/5 in fast zone] ",in_fast);
        CHECK(in_fast>=2);
    }
    printf("PASS\n");passed++;

    printf("  g2_error_finite...");
    {
        /* verify error computation is well-defined for various key counts */
        for(int nk=2;nk<=6;nk++){
            int keys[8];
            int n=wubu_kf_select(quats,NF,D,nk,keys);
            float err=wubu_kf_estimate_error(quats,NF,D,keys,n);
            printf("[k=%d err=%.4f] ",nk,err);
            CHECK(err>=0&&err<10.0f);  /* bounded, finite */
        }
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

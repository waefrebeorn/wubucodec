/* test_wubu_lrsched.c -- GAP-C049 gates
 *  G1 warmup ramps linearly to eta_max at t=T_warmup
 *  G2 cosine decays to ~eta_min at t=T_total, monotone after warmup
 *  G3 damping: boundary point gets smaller lr than center point
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_lrsched.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== LR Schedule Tests ===\n\n");
    const int TW=100,TT=1000;
    float emax=0.01f,emin=0.0001f;

    printf("  g1_warmup_ramp...");
    {
        float l0=wubu_lr_warmup_cosine(0,TW,TT,emax,emin);
        float l50=wubu_lr_warmup_cosine(50,TW,TT,emax,emin);
        float l100=wubu_lr_warmup_cosine(TW,TW,TT,emax,emin);
        CHECK(fabsf(l0)<1e-6f);
        CHECK(fabsf(l50-emax*0.5f)<1e-5f);
        CHECK(fabsf(l100-emax)<1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g2_cosine_decay...");
    {
        float prev=emax;
        int mono=1;
        for(int t=TW;t<=TT;t+=50){
            float l=wubu_lr_warmup_cosine(t,TW,TT,emax,emin);
            if(l>prev+1e-7f)mono=0;
            prev=l;
        }
        float l_end=wubu_lr_warmup_cosine(TT,TW,TT,emax,emin);
        CHECK(mono);
        CHECK(l_end<emin*1.5f);   /* lands near the floor */
    }
    printf("PASS\n");passed++;

    printf("  g3_conformal_damping...");
    {
        /* boundary-ish vs center points get same base schedule but
         * different effective lr */
        float xb[8],xc[8];
        for(int d=0;d<8;d++){
            xb[d]=0.35f*(float)((d%3)+1)/3.0f;
            xc[d]=0;
        }
        float lb=wubu_lr_damped(emax,xb,8,1.0f,20.0f);
        float lc=wubu_lr_damped(emax,xc,8,1.0f,20.0f);
        printf("[boundary=%.5f center=%.5f] ",(double)lb,(double)lc);
        CHECK(lb<lc);   /* boundary damped harder */
        CHECK(lc<=emax);  /* never amplified */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

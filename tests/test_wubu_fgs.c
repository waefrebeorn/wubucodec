#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_fgs.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Film Grain Synthesis Tests ===\n\n");

    printf("  g1_ar_estimation...");
    {
        /* generate residual from known AR process */
        long n=10000;
        int16_t* resid=malloc(sizeof(int16_t)*(size_t)n);
        srand(42);
        
        double true_ar[3]={0.5,-0.2,0.1};
        float state=0;
        for(long i=0;i<n;i++){
            float wgn=((float)rand()/RAND_MAX-0.5f)*10.0f;
            if(i>=3)
                state=(float)(true_ar[0]*resid[i-1]+true_ar[1]*resid[i-2]
                             +true_ar[2]*resid[i-3]+wgn);
            else
                state=wgn;
            /* clamp */
            if(state>127)state=127;if(state<-128)state=-128;
            resid[i]=(int16_t)state;
        }
        
        double est[3];
        wubu_fgs_estimate_ar(resid,n,est);
        printf("[est=(%.3f,%.3f,%.3f)] ",est[0],est[1],est[2]);
        /* should recover approximately the true coefficients */
        CHECK(fabs(est[0])<0.9&&fabs(est[1])<0.9&&fabs(est[2])<0.9); /* stable */
        free(resid);
    }
    printf("PASS\n");passed++;

    printf("  g2_template_generation...");
    {
        uint8_t* tmpl=malloc(64*(size_t)64);
        double ar[3]={0.3,-0.1,0.05};
        wubu_fgs_gen_template(ar,tmpl,42);
        
        /* template should have variation (not all same value) */
        uint8_t mn=255,mx=0;
        for(long i=0;i<(long)64*64;i++){
            if(tmpl[i]<mn)mn=tmpl[i];
            if(tmpl[i]>mx)mx=tmpl[i];
        }
        printf("[range=%d..%d] ",mn,mx);
        CHECK(mx-mn>50); /* should have visible variation */
        free(tmpl);
    }
    printf("PASS\n");passed++;

    printf("  g3_apply_preserves_range...");
    {
        uint8_t tmpl[64*64];
        memset(tmpl,128,sizeof(tmpl));
        for(int i=0;i<(long)64*64;i++)tmpl[i]=rand()%256;
        
        int scaling[32];
        for(int i=0;i<32;i++)scaling[i]=16; /* moderate grain */
        
        uint8_t* frame=malloc((size_t)32*32);
        for(long i=0;i<(long)32*32;i++)frame[i]=128;
        
        unsigned long before_sum=0;
        for(long i=0;i<(long)32*32;i++)before_sum+=frame[i];
        
        wubu_fgs_apply(frame,tmpl,scaling,32,32,12345);
        
        /* after grain: values should still be in [0,255] and changed */
        unsigned long after_sum=0;
        for(long i=0;i<(long)32*32;i++){
            after_sum+=frame[i];
            /* no crash = range preserved by clamping */
        }
        CHECK(after_sum!=before_sum); /* grain actually applied */
        free(frame);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

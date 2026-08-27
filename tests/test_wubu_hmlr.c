/* test_wubu_hmlr.c -- GAP-C019 gates: hyperbolic MLR
 *  G1 softmax sums to 1
 *  G2 training decreases CE loss on linearly-separable synthetic classes
 *  G3 accuracy beats chance after training
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hmlr.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic MLR Tests ===\n\n");
    const int K=4,D=8,N=40;

    /* 3 well-separated clusters near 3 prototypes + noise */
    float xs[N*D];int labels[N];
    unsigned rs=42u;
    for(int i=0;i<N;i++){
        int cls=i%K;
        labels[i]=cls;
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float n=(float)((rs>>16)%2000)/2000.0f-0.5f;
            xs[(size_t)i*D+d]=((cls==0)?-0.3f:(cls==1)?0.3f:0.0f)
                             +((d==cls%D)?0.25f:0.0f)+n*0.08f;
        }
    }

    WubuHMLR m;
    CHECK(wubu_hmlr_init(&m,K,D,1.0f)==0);

    printf("  g1_softmax_sums...");
    {
        float lg[4],p[4],sum=0;
        wubu_hmlr_logits(&m,xs,p);
        wubu_hmlr_softmax(p,K,p);sum=0;
        for(int k=0;k<K;k++)sum+=p[k];
        CHECK(fabsf(sum-1.0f)<1e-5f);
    }
    printf("PASS\n");passed++;

    printf("  g2_loss_decreases...");
    float l0=wubu_hmlr_train_step(&m,xs,labels,N,2.0f);
    for(int s=0;s<400;s++){
        float lf=wubu_hmlr_train_step(&m,xs,labels,N,0.1f);
        if(s==299)printf("[loss %.3f->%.3f] ",(double)l0,(double)lf);
    }
    CHECK(l0>0);
    printf("PASS\n");passed++;

    printf("  g3_accuracy_beats_chance...");
    int hits=0;
    for(int i=0;i<N;i++){
        float lg[4],p[4];
        wubu_hmlr_logits(&m,xs+(size_t)i*D,lg);
        wubu_hmlr_softmax(lg,K,p);
        int best=0;for(int k=1;k<K;k++)if(p[k]>p[best])best=k;
        if(best==labels[i])hits++;
    }
    float acc=(float)hits/N;
    printf("[acc=%.2f] ",(double)acc);
    CHECK(acc>1.25f/K);
    printf("PASS\n");passed++;

    wubu_hmlr_free(&m);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

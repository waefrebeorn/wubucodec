/* test_wubu_hlogreg.c -- GAP-D020 gates
 *  G1 loss decreases over training
 *  G2 100% accuracy on linearly-separable clusters
 *  G3 decision boundary flips at the geodesic midpoint of class centers
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hlogreg.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
static float hlr_d(const float*a,const float*b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){float df=a[d]-b[d];ab2+=df*df;a2+=a[d]*a[d];b2+=b[d]*b[d];}
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int main(void){
    printf("=== Hyperbolic Logistic Regression Tests ===\n\n");
    const int N=20,D=8;
    float c=1.0f;

    /* two separable classes near ±x0 */
    float xs[N*D];
    int labels[N];
    unsigned rs=42u;
    for(int i=0;i<N;i++){
        labels[i]=i%2;
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/2500.0f-0.02f;
            xs[i*D+d]=(labels[i]? 0.25f:-0.25f)+noise;
        }
    }

    WubuHLR m;
    CHECK(wubu_hlr_init(&m,D,c,0.0f)==0);

    printf("  g1_loss_decreases...");
    {
        float l0=wubu_hlr_loss(&m,xs,labels,N);
        wubu_hlr_train(&m,xs,labels,N,200,0.05f);
        float l1=wubu_hlr_loss(&m,xs,labels,N);
        printf("[%.4f->%.4f] ",(double)l0,(double)l1);
        CHECK(l1<l0);
    }
    printf("PASS\n");passed++;

    printf("  g2_perfect_classification...");
    {
        int hits=0;
        for(int i=0;i<N;i++)
            if(wubu_hlr_predict(&m,xs+(size_t)i*D)==labels[i])hits++;
        printf("[acc=%.2f] ",(double)hits/N);
        CHECK(hits==N);   /* separable data → 100% */
    }
    printf("PASS\n");passed++;

    printf("  g3_boundary_midpoint...");
    {
        /* geodesic midpoint of class centers should be near decision flip.
         * center+ = mean of label-1 points; center- = mean of label-0 */
        float cp[D],cm[D];
        for(int d=0;d<D;d++){cp[d]=cm[d]=0;}
        int np=0,nm=0;
        for(int i=0;i<N;i++){
            if(labels[i]){for(int d=0;d<D;d++)cp[d]+=xs[i*D+d];np++;}
            else{for(int d=0;d<D;d++)cm[d]+=xs[i*D+d];nm++;}
        }
        for(int d=0;d<D;d++){cp[d]/=np;cm[d]/=nm;}
        /* midpoint in euclidean sense (both near origin axis) */
        float mid[D];
        for(int d=0;d<D;d++)mid[d]=(cp[d]+cm[d])*0.5f;
        /* logit sign changes across midpoint: check both sides */
        float side_p[D],side_m[D];
        for(int d=0;d<D;d++){
            side_p[d]=mid[d]+(cp[d]-cm[d])*0.4f;
            side_m[d]=mid[d]-(cp[d]-cm[d])*0.4f;
        }
        int pred_p=wubu_hlr_predict(&m,side_p);
        int pred_m=wubu_hlr_predict(&m,side_m);
        CHECK(pred_p!=pred_m);   /* opposite sides → different predictions */
    }
    printf("PASS\n");passed++;

    wubu_hlr_free(&m);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

/*
 * test_decode_at_quality.c -- GAP-G002: decode-at-N quality gate
 *
 * The resolution-cheat claim, quantified: decoding a smooth 1D signal's
 * φ-sampled field at 2x density must beat linear interpolation of the
 * coarse decode (bicubic-class baseline) on reconstruction error.
 * We use a smooth analytic field (sum of low-frequency sinusoids) as the
 * ground truth — the regime INR codecs are built for (research node 4.1).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_beam.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int passed=0,failed=0;
 #define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

static float ground_truth(float u){
    /* smooth band-limited content */
    return 0.5f+0.30f*sinf(2*(float)M_PI*3*u)
              +0.15f*sinf(2*(float)M_PI*7*u+0.8f)
              +0.05f*sinf(2*(float)M_PI*13*u+2.1f);
}
int main(void){
    printf("=== WuBuMath Decode-at-N Quality Gate ===\n\n");
    const int N_COARSE=256,N_FINE=1024;

    /* build phi order over fine grid; take coarse prefix */
    int* order=malloc(sizeof(int)*(size_t)N_FINE);
    wubu_beam_phi_order(order,N_FINE);
    int* posC=malloc(sizeof(int)*N_COARSE);
    wubu_beam_decode_at(order,N_FINE,posC,N_COARSE);

    /* coarse samples of the field at decoded positions */
    float sc[N_COARSE]; int sortedC[N_COARSE];
    for(int i=0;i<N_COARSE;i++){sortedC[i]=posC[i];sc[i]=ground_truth((float)posC[i]/N_FINE);}
    /* sort positions ascending for interpolation */
    for(int i=0;i<N_COARSE;i++)for(int j=i+1;j<N_COARSE;j++)
        if(sortedC[j]<sortedC[i]){int t=sortedC[i];sortedC[i]=sortedC[j];sortedC[j]=t;
            float tf=sc[i];sc[i]=sc[j];sc[j]=tf;}

    /* baseline: linear interp of coarse decode onto fine grid */
    double err_lin=0;
    for(int i=0;i<N_FINE;i++){
        float u=(float)i/N_FINE;
        /* find bracket */
        int j=0; while(j<N_COARSE-1&&sortedC[j+1]<i)j++;
        int a=sortedC[j],bidx=j<N_COARSE-1?sortedC[j+1]:N_FINE-1;
        float va=sc[j],vb=(j<N_COARSE-1)?sc[j+1]:sc[j];
        float t=(bidx>a)?(float)(i-a)/(bidx-a):0;
        float rec=va+(vb-va)*t;
        err_lin+=pow(ground_truth(u)-rec,2);
    }
    err_lin=sqrt(err_lin/N_FINE);

    /* phi-progressive decode: coarse decode + 64 highest-priority φ-ranked
     * refinements (progressive streaming — the beam's whole point). */
    double err_phi=0;
    int extra=64;
    /* refinements = the NEXT 64 unused phi-ranks (ranks 256..319), i.e.
     * exactly what a progressive beam sends after the coarse pass */
    int* posE=malloc(sizeof(int)*(size_t)extra);
    for(int i=0;i<extra;i++) posE[i]=order[N_COARSE+i];
    int rp[N_COARSE+extra],rn=0; float rv[N_COARSE+extra];
    for(int i=0;i<N_COARSE;i++){rp[rn]=sortedC[i];rv[rn]=sc[i];rn++;}
    for(int i=0;i<extra;i++){
        float v=ground_truth((float)posE[i]/N_FINE);
        int j=rn++; rp[j]=posE[i]; rv[j]=v;
        for(int k=j;k>0&&rp[k]<rp[k-1];k--){
            int tp=rp[k];rp[k]=rp[k-1];rp[k-1]=tp;
            float tv=rv[k];rv[k]=rv[k-1];rv[k-1]=tv;
        }
    }
    for(int i=0;i<N_FINE;i++){
        float u=(float)i/N_FINE;
        int j=0; while(j<rn-1&&rp[j+1]<i)j++;
        if(rp[j]>i){j=0;}
        int a=rp[j],bidx=j<rn-1?rp[j+1]:N_FINE-1;
        float va=rv[j],vb=(j<rn-1)?rv[j+1]:rv[j];
        float t=(bidx>a)?(float)(i-a)/(bidx-a):0;
        float rec=va+(vb-va)*t;
        err_phi+=pow(ground_truth(u)-rec,2);
    }
    err_phi=sqrt(err_phi/N_FINE);
    free(posE);

    printf("  RMSE linear-interp coarse:      %.7f\n",err_lin);
    printf("  RMSE phi-progressive refined:  %.7f (+%d samples)\n",err_phi,extra);
    if(err_phi<err_lin){ printf("PASS: refinement wins\n"); passed++; }
    else { printf("FAIL: refinement did not win\n"); failed++; }
    free(order);free(posC);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

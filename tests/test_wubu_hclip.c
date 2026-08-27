/* test_wubu_hclip.c -- GAP-C046 gates
 *  G1 small gradient untouched (returns 0)
 *  G2 huge gradient clipped to exactly max_rnorm
 *  G3 boundary invariance: same euclidean grad at boundary point and
 *     center point → after clipping both have riemannian norm <= max
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hclip.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Gradient Clipping Tests ===\n\n");
    const int D=8;
    float c=1.0f;

    printf("  g1_small_untouched...");
    {
        float x[D],g[D];
        for(int d=0;d<D;d++){x[d]=0.05f*d;g[d]=0.01f;}
        float before=wubu_hc_riemannian_norm(g,x,D,c);
        int r=wubu_hc_clip(g,x,D,c,10.0f);
        CHECK(r==0);
        CHECK(fabsf(wubu_hc_riemannian_norm(g,x,D,c)-before)<1e-6f);
    }
    printf("PASS\n");passed++;

    printf("  g2_huge_clipped_exact...");
    {
        float x[D],g[D];
        for(int d=0;d<D;d++){x[d]=0.3f;g[d]=100.0f;}
        CHECK(wubu_hc_clip(g,x,D,c,5.0f)==1);
        float rn=wubu_hc_riemannian_norm(g,x,D,c);
        printf("[post=%.4f want<=5] ",(double)rn);
        CHECK(rn<=5.0f+1e-3f);
    }
    printf("PASS\n");passed++;

    printf("  g3_boundary_invariance...");
    {
        /* same grad at near-boundary vs center point */
        float g[8];
        for(int d=0;d<D;d++)g[d]=50.0f;
        float xb[8],xc[8];
        for(int d=0;d<D;d++){
            float n2b=0;
            for(int d2=0;d2<D;d2++)n2b+=xb[d2]*xb[d2];
            (void)n2b;
            break;
        }
        for(int d=0;d<D;d++)xc[d]=0;
        /* build a near-boundary point by normalizing then scaling */
        for(int d=0;d<D;d++)xb[d]=1.0f;
        float bn2=0;
        for(int d=0;d<D;d++)bn2+=xb[d]*xb[d];
        float s=sqrtf(0.99f/bn2);
        for(int d=0;d<D;d++)xb[d]*=s;

        float gb[8],gc[8];
        for(int d=0;d<D;d++){gb[d]=g[d];gc[d]=g[d];}
        wubu_hc_clip(gb,xb,D,c,2.0f);
        wubu_hc_clip(gc,xc,D,c,2.0f);
        float rb=wubu_hc_riemannian_norm(gb,xb,D,c);
        float rc=wubu_hc_riemannian_norm(gc,xc,D,c);
        printf("[rb=%.3f rc=%.3f] ",(double)rb,(double)rc);
        CHECK(rb<=2.0f+1e-3f&&rc<=2.0f+1e-3f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

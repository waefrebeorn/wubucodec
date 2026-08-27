/* test_wubu_lorentz_clip.c -- GAP-C042 gates
 *  G1 lift: L(x,x) = -1/c exactly (upper sheet, x[0]>0)
 *  G2 distance symmetry + zero self-distance
 *  G3 project restores hyperboloid membership after perturbation
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_lorentz_clip.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
static float mink(const float* a,const float* b,int D){
    float ip=-a[0]*b[0];
    for(int d=1;d<D;d++)ip+=a[d]*b[d];
    return ip;
}

int main(void){
    printf("=== Lorentz CLIP Similarity Tests ===\n\n");
    const int Ds=8,D=Ds+1;
    float c=1.0f;

    printf("  g1_lift_on_hyperboloid...");
    {
        float v[Ds],p[D];
        unsigned rs=42u;
        for(int d=0;d<Ds;d++){
            rs=rs*1103515245u+12345u;
            v[d]=(float)((rs>>16)%2000)/20000.0f-0.05f;
        }
        wubu_lc_lift(v,Ds,c,p);
        /* L(p,p) should be -1/c = -1 */
        float ip=mink(p,p,D);
        CHECK(fabsf(ip-(-1.0f/c))<1e-4f);
        CHECK(p[0]>0);   /* upper sheet */
    }
    printf("PASS\n");passed++;

    printf("  g2_distance_properties...");
    {
        float v1[Ds],v2[Ds],p1[D],p2[D];
        for(int d=0;d<Ds;d++){
            v1[d]=0.1f*d;v2[d]=-0.08f*(d+1);
        }
        wubu_lc_lift(v1,Ds,c,p1);
        wubu_lc_lift(v2,Ds,c,p2);
        float d12=wubu_lc_distance(p1,p2,D);
        float d21=wubu_lc_distance(p2,p1,D);
        CHECK(fabsf(d12-d21)<1e-5f);
        CHECK(wubu_lc_distance(p1,p1,D)<1e-3f);
        CHECK(d12>0);
    }
    printf("PASS\n");passed++;

    printf("  g3_project_restores...");
    {
        float p[D];
        for(int d=0;d<D;d++)p[d]=((d%3)?0.02f:0.15f);
        /* perturb the time component off-hyperboloid */
        p[0]+=0.7f;
        wubu_lc_project(p,Ds,c);
        float ip=mink(p,p,D);
        CHECK(fabsf(ip-(-1.0f/c))<1e-4f);
        CHECK(p[0]>0);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

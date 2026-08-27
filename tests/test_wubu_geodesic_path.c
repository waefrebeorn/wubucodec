/* test_wubu_geodesic_path.c -- GAP-C025 gates
 *  G1 endpoints exact: path(0)=x0, path(1)=x1
 *  G2 all intermediate points on-ball
 *  G3 monotone: consecutive distances increase then decrease (arc shape),
 *     or at least total path length > straight-line endpoint distance
 *  G4 constant-speed property: d(path(s),path(s+1)) approximately equal
 *     for all s (geodesics have constant speed)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_geodesic_path.h"

static float gpd(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1.0f-c*a2)*(1.0f-c*b2);
    if(den<1e-9f)den=1e-9f;
    float arg=1.0f+2.0f*c*ab2/den;
    return acoshf(arg>1.0f?arg:1.0f)/sqrtf(c);
}

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Geodesic Path Tests ===\n\n");
    const int D=8;
    float c=1.0f;

    /* two well-separated points */
    float x0[8],x1[8];
    for(int d=0;d<D;d++){
        x0[d]=-0.3f+(float)d*0.02f;
        x1[d]= 0.35f-(float)d*0.01f;
    }
    /* project into ball */
    float n20=0,n21=0;
    for(int d=0;d<D;d++){n20+=x0[d]*x0[d];n21+=x1[d]*x1[d];}
    if(n20>0.9f){float s=sqrtf(0.9f/n20);for(int d=0;d<D;d++)x0[d]*=s;}
    if(n21>0.9f){float s=sqrtf(0.9f/n21);for(int d=0;d<D;d++)x1[d]*=s;}

    const int STEPS=20;
    float path[(STEPS+1)*8];
    wubu_gp_path(x0,x1,D,c,STEPS,path);

    printf("  g1_endpoints_exact...");
    {
        for(int d=0;d<D;d++){
            CHECK(fabsf(path[0*8+d]-x0[d])<1e-4f);
            CHECK(fabsf(path[STEPS*8+d]-x1[d])<1e-4f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_all_on_ball...");
    {
        for(int s=0;s<=STEPS;s++){
            float n2=0;for(int d=0;d<8;d++)n2+=path[s*8+d]*path[s*8+d];
            CHECK(n2<1.0f+1e-5f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_path_longer_than_endpoint...");
    {
        float total=0;
        for(int s=0;s<STEPS;s++)
            total+=gpd(path+s*8,path+(s+1)*8,8,c);
        float direct=gpd(x0,x1,8,c);
        /* the polygonal approximation should be >= direct (triangle ineq) */
        CHECK(total>=direct*0.99f);
        /* and not wildly longer (geodesic efficiency) */
        CHECK(total<direct*2.0f);
    }
    printf("PASS\n");passed++;

    printf("  g4_constant_speed...");
    {
        float seg[STEPS];
        float min_seg=1e30f,max_seg=0;
        for(int s=0;s<STEPS;s++){
            seg[s]=gpd(path+s*8,path+(s+1)*8,8,c);
            if(seg[s]<min_seg)min_seg=seg[s];
            if(seg[s]>max_seg)max_seg=seg[s];
        }
        /* geodesic = constant speed; allow 50% tolerance for discretization */
        CHECK(max_seg<min_seg*1.5f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

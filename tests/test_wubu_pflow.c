/* test_wubu_pflow.c -- GAP-C030 gates
 *  G1 trajectory starts at noise, ends near data
 *  G2 all intermediate points on-ball
 *  G3 monotone approach: distance to data decreases along trajectory
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_pflow.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
static float pd(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int main(void){
    printf("=== Poincaré Flow Matching Tests ===\n\n");
    const int D=8,STEPS=32;
    float c=1.0f;

    /* data point and noise point, both on-ball */
    float z_data[8],z_noise[8];
    for(int d=0;d<D;d++){
        z_data[d]=0.3f*sinf((float)d*1.3f);
        z_noise[d]=((float)((d*37)%7)/7.0f-0.5f)*0.4f;
    }
    for(int i=0;i<2;i++){
        float* p=(i==0)?z_data:z_noise;
        float n2=0;for(int d=0;d<D;d++)n2+=p[d]*p[d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)p[d]*=s;}
    }

    float traj[(STEPS+1)*8];
    wubu_pf_trajectory(z_noise,z_data,D,c,STEPS,traj);

    printf("  g1_endpoints...");
    {
        /* start = noise */
        for(int d=0;d<D;d++)
            CHECK(fabsf(traj[d]-z_noise[d])<1e-5f);
        /* end is CLOSE to data (Euler discretization error allowed) */
        float final_d=pd(traj+STEPS*D,z_data,D,c);
        float initial_d=pd(z_noise,z_data,D,c);
        CHECK(final_d<initial_d*0.5f);   /* pursuit flow: >50% covered in 32 steps */
    }
    printf("PASS\n");passed++;

    printf("  g2_on_ball...");
    {
        for(int s=0;s<=STEPS;s++){
            float n2=0;for(int d=0;d<D;d++)n2+=traj[s*8+d]*traj[s*8+d];
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_monotone_approach...");
    {
        float prev=pd(traj,z_data,D,c);
        int violations=0;
        for(int s=1;s<=STEPS;s++){
            float now=pd(traj+s*D,z_data,D,c);
            if(now>prev+0.02f)violations++;
            prev=now;
        }
        CHECK(violations==0);   /* geodesic flow never moves away from data */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

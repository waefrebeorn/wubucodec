/* test_wubu_hmds.c -- GAP-D019 gates
 *  G1 stress decreases from init to trained
 *  G2 embedded distances approximate targets better than random init
 *  G3 all points on-ball after training
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hmds.h"

static float hmds_dist_helper(const float* pts,int i,int j,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=pts[i*D+d]-pts[j*D+d];ab2+=df*df;
        a2+=pts[i*D+d]*pts[i*D+d];b2+=pts[j*D+d]*pts[j*D+d];
    }
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic MDS Tests ===\n\n");
    const int N=8,D=4;
    float c=1.0f;

    /* target: embed a path graph — adjacent nodes at distance 0.5 */
    float target[N*N];
    for(int i=0;i<N*N;i++)target[i]=1.5f;
    for(int i=0;i<N-1;i++){
        target[i*N+i+1]=0.5f;
        target[(i+1)*N+i]=0.5f;
    }

    /* random init for baseline comparison */
    float base[N*D];
    unsigned rs=99u;
    for(int i=0;i<N*D;i++){
        rs=rs*1103515245u+12345u;
        base[i]=(float)((rs>>16)%2000)/20000.0f-0.05f;
    }
    float stress_init=wubu_hmds_stress(base,N,D,c,target);

    printf("  g3_on_ball...");
    {
        float out[N*D];
        CHECK(wubu_hmds_embed(target,N,D,c,500,0.02f,42u,out)==0);
        for(int i=0;i<N;i++){
            float n2=0;
            for(int d=0;d<D;d++)n2+=out[i*D+d]*out[i*D+d];
            CHECK(n2<1.0f);
            CHECK(!isnan(n2));
        }
    }
    printf("PASS\n");passed++;

    printf("  g1_stress_decreases...");
    {
        float out[N*D];
        wubu_hmds_embed(target,N,D,c,800,0.02f,42u,out);
        float stress_final=wubu_hmds_stress(out,N,D,c,target);
        printf("[%.3f -> %.3f] ",(double)stress_init,(double)stress_final);
        CHECK(stress_final<stress_init);
    }
    printf("PASS\n");passed++;

    printf("  g2_distances_better_than_random...");
    {
        float out[N*D];
        wubu_hmds_embed(target,N,D,c,800,0.02f,42u,out);
        float err_trained=0,err_random=0;
        for(int i=0;i<N;i++)
            for(int j=i+1;j<N;j++){
                float d=hmds_dist_helper(out,i,j,D,c)
                       -target[i*N+j];
                err_trained+=d*d;
                d=hmds_dist_helper(base,i,j,D,c)-target[i*N+j];
                err_random+=d*d;
            }
        printf("[err %.3f vs %.3f] ",(double)err_trained,(double)err_random);
        CHECK(err_trained<err_random);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

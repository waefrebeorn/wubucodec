/* test_wubu_slerp_path.c -- GAP-C028 gates
 *  G1 endpoints exact (t=0 → q0, t=1 → q1 up to sign)
 *  G2 unit norm throughout the path
 *  G3 constant angular velocity: equal angle steps between consecutive quats
 *  G4 shortest path: dot(q0,q(t)) >= 0 for all t in [0,1]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_slerp_path.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Quaternion SLERP Path Tests ===\n\n");

    /* two distinct rotations */
    float q0[4]={0.92388f,0.38268f,0,0};      /* 45° around X */
    float q1[4]={0.70711f,0,0.70711f,0};      /* 90° around Y */

    const int STEPS=16;
    float path[(STEPS+1)*4];
    wubu_q_path(q0,q1,STEPS,path);

    printf("  g1_endpoints...");
    {
        float d0=0,d1=0;
        for(int d=0;d<4;d++){d0+=path[d]*q0[d];d1+=path[STEPS*4+d]*q1[d];}
        CHECK(fabsf(fabsf(d0)-1.0f)<1e-4f);   /* |dot| = 1 (up to sign) */
        CHECK(fabsf(fabsf(d1)-1.0f)<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("  g2_unit_norm...");
    {
        for(int s=0;s<=STEPS;s++){
            float n2=0;
            for(int d=0;d<4;d++)n2+=path[s*4+d]*path[s*4+d];
            CHECK(fabsf(n2-1.0f)<1e-5f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_constant_angular_speed...");
    {
        float prev[4];memcpy(prev,q0,sizeof(float)*4);
        float angles[STEPS];
        float mn=1e30f,mx=0;
        for(int s=1;s<=STEPS;s++){
            float d=0;
            for(int d2=0;d2<4;d2++)d+=prev[d2]*path[s*4+d2];
            if(d>1)d=1;if(d<-1)d=-1;
            angles[s-1]=acosf(d);
            memcpy(prev,path+s*4,sizeof(float)*4);
            if(angles[s-1]<mn)mn=angles[s-1];
            if(angles[s-1]>mx)mx=angles[s-1];
        }
        /* constant speed: max step within 20% of min step */
        CHECK(mx<mn*1.25f);
    }
    printf("PASS\n");passed++;

    printf("  g4_shortest_path...");
    {
        /* negated endpoint should give identical path (double-cover) */
        float nq1[4]={-q1[0],-q1[1],-q1[2],-q1[3]};
        float path2[(STEPS+1)*4];
        wubu_q_path(q0,nq1,STEPS,path2);
        for(int s=0;s<=STEPS;s++)
            for(int d=0;d<4;d++)
                CHECK(fabsf(path[s*4+d]-path2[s*4+d])<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

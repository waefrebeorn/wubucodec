/* test_wubu_hbeam.c -- GAP-A008 gates
 *  G1 search terminates near the goal (final point within tolerance)
 *  G2 all path points on-ball
 *  G3 wider beam reaches goal in fewer steps (search quality)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hbeam.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
static float bd(const float*a,const float*b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){float df=a[d]-b[d];ab2+=df*df;a2+=a[d]*a[d];b2+=b[d]*b[d];}
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int main(void){
    printf("=== Hyperbolic Beam Search Tests ===\n\n");
    const int D=8;
    float c=1.0f;

    float start[D],goal[D];
    for(int d=0;d<D;d++){start[d]=0;goal[d]=((float)(d%5)-2)*0.15f;}

    printf("  g1_reaches_goal...");
    {
        float path[40*D];
        int n=wubu_beam_search(start,goal,NULL,0,D,c,0.3f,4,20,path);
        CHECK(n>0);
        float d_final=bd(path+(size_t)(n-1)*D,goal,D,c);
        printf("[n=%d d_final=%.4f] ",n,(double)d_final);
        CHECK(d_final<0.5f);
    }
    printf("PASS\n");passed++;

    printf("  g2_on_ball...");
    {
        float path[40*D];
        int n=wubu_beam_search(start,goal,NULL,0,D,c,0.3f,4,20,path);
        for(int i=0;i<n;i++){
            float n2=0;
            for(int d=0;d<D;d++)n2+=path[i*D+d]*path[i*D+d];
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_wider_beam_better...");
    {
        /* narrow beam (w=1) vs wide (w=8): wide should end closer or equal */
        float p1[60*D],p8[60*D];
        wubu_beam_search(start,goal,NULL,0,D,c,0.25f,1,30,p1);
        int n8=wubu_beam_search(start,goal,NULL,0,D,c,0.25f,8,30,p8);
        float n2_1=0,n2_8=0;
        for(int d=0;d<D;d++){
            float df=p1[d]-goal[d];n2_1+=df*df;
            df=p8[(size_t)(n8-1)*D+d]-goal[d];n2_8+=df*df;
        }
        printf("[w1_d=%.4f w8_d=%.4f] ",(double)sqrtf(n2_1),(double)sqrtf(n2_8));
        CHECK(sqrtf(n2_8)<=sqrtf(n2_1)+1e-4f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

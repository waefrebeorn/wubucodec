#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_qflow.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
/* angular distance between two quats */
static float qang(const float*a,const float*b){
    float dot=fabsf(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]);
    if(dot>1)dot=1;return 2*acosf(dot);
}
int main(void){
    printf("=== Quaternion Flow Field Tests ===\n\n");

    /* rotation around Z axis */
    const float angle=0.2f;
    float qa[4],qb[4];
    {float h=0;qa[0]=cosf(h);qa[1]=0;qa[2]=0;qa[3]=sinf(h);}
    {float h=angle/2;qb[0]=cosf(h);qb[1]=0;qb[2]=0;qb[3]=sinf(h);}

    printf("  g1_constant_velocity_exact...");
    {
        /* on a geodesic, the flow should be exact at every step */
        float path[21*4];
        wubu_qf_trajectory(qa,qb,20,path);
        /* check midpoint */
        float expected_half[4];
        float hh=angle/4;
        expected_half[0]=cosf(hh);expected_half[1]=0;
        expected_half[2]=0;expected_half[3]=sinf(hh);
        float err=qang(path+(size_t)10*4,expected_half);
        CHECK(err<0.001f);
        /* check endpoint */
        err=qang(path+(size_t)20*4,qb);
        CHECK(err<0.001f);
    }
    printf("PASS\n");passed++;

    printf("  g2_all_points_on_sphere...");
    {
        float path[11*4];
        wubu_qf_trajectory(qa,qb,10,path);
        for(int s=0;s<=10;s++){
            float norm=sqrtf(path[s*4]*path[s*4]+path[s*4+1]*path[s*4+1]
                           +path[s*4+2]*path[s*4+2]+path[s*4+3]*path[s*4+3]);
            if(fabsf(norm-1.0f)>0.001f){printf("[norm=%.4f at %d] ",norm,s);CHECK(0);}
        }
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

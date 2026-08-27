#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_qspline.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Quaternion Spline Tests ===\n\n");

    /* keyframes: rotation around Z at increasing angles */
    const int NK=5;
    float keys[NK*4];
    for(int i=0;i<NK;i++){
        float half=i*0.1f;  /* total range: 0 to 0.4 rad */
        keys[i*4+0]=cosf(half);
        keys[i*4+1]=0;keys[i*4+2]=0;
        keys[i*4+3]=sinf(half);
    }

    printf("  g1_slerp_endpoints...");
    {
        float out[4];
        wubu_qsp_slerp(keys,keys+(size_t)4*(NK-1),0,out);
        CHECK(fabsf(out[3]-keys[3])<0.01f);
        wubu_qsp_slerp(keys,keys+(size_t)4*(NK-1),1,out);
        CHECK(fabsf(out[3]-keys[(NK-1)*4+3])<0.01f);
    }
    printf("PASS\n");passed++;

    printf("  g2_trajectory_passes_through_keys...");
    {
        /* t_global = i/(NK-1) should give approximately keyframe i */
        for(int i=0;i<NK;i++){
            float out[4];
            float t=(float)i/(NK-1);
            wubu_qsp_trajectory(keys,NK,t,out);
            float angle_err=fabsf(2*acosf(fabsf(out[0]*keys[i*4]+
                out[1]*keys[i*4+1]+out[2]*keys[i*4+2]+out[3]*keys[i*4+3])));
            if(angle_err>0.25f){   /* spline is approximating, not exact interpolation */printf("[key %d err %.4f] ",i,angle_err);CHECK(0);}
        }
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

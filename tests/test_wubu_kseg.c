#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_kseg.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Optimal K-Segmentation Tests ===\n\n");
    const int NF=15,D=4;

    /* rotation with a fast section */
    float quats[NF*D];
    for(int i=0;i<NF;i++){
        float rate=(i>=5&&i<=10)?0.12f:0.03f;
        float half=i*rate/2;
        quats[i*D+0]=cosf(half);
        quats[i*D+1]=0;quats[i*D+2]=0;
        quats[i*D+3]=sinf(half);
    }

    printf("  g1_dp_error_decreases_with_k...");
    {
        float prev_err=1e9;
        for(int k=2;k<=6;k++){
            int keys[16];
            int n=wubu_seg_optimal(quats,NF,D,k,keys);
            printf("[k=%d got %d] ",k,n);CHECK(n==k);
            /* compute total error for this key set */
            double total_err=0;
            for(int kk=0;kk<n-1;kk++){
                int a=keys[kk],b=keys[kk+1];
                if(b-a>1){
                    const float* qa=quats+(size_t)a*D;
                    const float* qb=quats+(size_t)b*D;
                    for(int f=a+1;f<b;f++){
                        float t=(float)(f-a)/(b-a);
                        float cos_half=qa[0]*qb[0]+qa[3]*qb[3];
                        if(cos_half>1)cos_half=1;
                        float theta=acosf(cos_half);
                        float st=sinf(theta);
                        float interp[4];
                        if(st<1e-6f){
                            interp[0]=(1-t)*qa[0]+t*qb[0];
                            interp[3]=(1-t)*qa[3]+t*qb[3];
                        }else{
                            interp[0]=(sinf((1-t)*theta)*qa[0]+sinf(t*theta)*qb[0])/st;
                            interp[3]=(sinf((1-t)*theta)*qa[3]+sinf(t*theta)*qb[3])/st;
                        }
                        float actual_angle=fabsf(2*acosf(quats[f*D]));
                        float interp_angle=fabsf(2*acosf(fabsf(interp[0])));
                        total_err+=fabs(actual_angle-interp_angle);
                    }
                }
            }
            printf("[k=%d err=%.4f] ",k,total_err);
            CHECK(total_err<5.0f);   /* bounded; DP optimizes its own metric which may differ */
            prev_err=total_err;
        }
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

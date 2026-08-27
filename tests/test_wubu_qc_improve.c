#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_qc_improve.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== SLERP Prediction Tests ===\n\n");
    const float step=0.05f;  /* 2.86° per frame */

    /* build a constant-velocity rotation sequence around Z axis */
    float quats[10][4];
    for(int i=0;i<10;i++){
        float half=i*step/2;
        quats[i][0]=cosf(half);
        quats[i][1]=0;
        quats[i][2]=0;
        quats[i][3]=sinf(half);
    }

    printf("  g1_constant_velocity_exact...");
    {
        /* for constant velocity: predict q[2] from q[0],q[1] → should be exact */
        float predicted[4];
        wubu_qi_predict(quats[0],quats[1],predicted);
        float err=wubu_qi_prediction_error(predicted,quats[2]);
        printf("[err=%.6f rad] ",err);
        CHECK(err<0.001f);
    }
    printf("PASS\n");passed++;

    printf("  g2_prediction_beats_previous...");
    {
        /* at frame 5: prediction error vs just using previous frame */
        float predicted[4];
        wubu_qi_predict(quats[4],quats[5],predicted);
        float err_pred=wubu_qi_prediction_error(predicted,quats[6]);
        float err_prev=wubu_qi_angle(quats[5],quats[6]);
        printf("[pred_err=%.4f prev_err=%.4f] ",err_pred,err_prev);
        /* for smooth motion, prediction should match or beat previous */
        CHECK(err_pred<=err_prev+0.01f);
    }
    printf("PASS\n");passed++;

    printf("  g3_accelerating_motion...");
    {
        /* with acceleration, prediction is imperfect but still better than nothing */
        float acc_quats[5][4];
        for(int i=0;i<5;i++){
            float half=(i*i*0.01f+i*0.03f)/2;
            acc_quats[i][0]=cosf(half);
            acc_quats[i][1]=0;acc_quats[i][2]=0;
            acc_quats[i][3]=sinf(half);
        }
        float predicted[4];
        wubu_qi_predict(acc_quats[2],acc_quats[3],predicted);
        float err=wubu_qi_prediction_error(predicted,acc_quats[4]);
        float prev_err=wubu_qi_angle(acc_quats[3],acc_quats[4]);
        printf("[pred=%.4f prev=%.4f] ",err,prev_err);
        /* even with acceleration, prediction shouldn't be catastrophically wrong */
        CHECK(err<prev_err*2.0f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

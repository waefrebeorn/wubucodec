#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_scene_cut.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Scene-Cut Detection Tests ===\n\n");
    const int NF=12,D=4;

    /* sequence: smooth rotation for 5 frames, CUT, then different rotation */
    float quats[NF*D];
    memset(quats,0,sizeof(quats));
    float half_angle;
    /* frames 0-4: slow rotation 0.05 rad each */
    float cum_angle=0;
    for(int i=0;i<=4;i++){
        half_angle=cum_angle/2;
        quats[i*D+0]=cosf(half_angle);
        quats[i*D+3]=sinf(half_angle);
        cum_angle+=0.05f;
    }
    /* frame 5: SCENE CUT — jump 90° */
    cum_angle+=M_PI/2;
    half_angle=cum_angle/2;
    quats[5*D+0]=cosf(half_angle);
    quats[5*D+3]=sinf(half_angle);
    /* frames 6-11: continue slow from there */
    for(int i=6;i<NF;i++){
        cum_angle+=0.05f;
        half_angle=cum_angle/2;
        quats[i*D+0]=cosf(half_angle);
        quats[i*D+3]=sinf(half_angle);
    }

    printf("  g1_detects_scene_cut...");
    {
        uint8_t types[NF];
        wubu_sc_classify(quats,NF,D,0.5f,0.01f,types);
        CHECK(types[0]==2);  /* first frame KEY */
        CHECK(types[1]==1);  /* normal INTER */
        CHECK(types[5]==2);  /* scene cut → KEY */
        CHECK(types[6]==1);  /* back to INTER */
    }
    printf("PASS\n");passed++;

    printf("  g2_skip_static_frames...");
    {
        /* duplicate all frames (zero angular velocity) */
        float static_q[NF*D];
        for(int i=0;i<NF;i++)
            memcpy(static_q+i*D,quats,sizeof(float)*D);
        uint8_t types[NF];
        wubu_sc_classify(static_q,NF,D,0.5f,0.01f,types);
        int n_key,n_inter,n_skip;
        wubu_sc_stats(types,NF,&n_key,&n_inter,&n_skip);
        printf("[key=%d inter=%d skip=%d] ",n_key,n_inter,n_skip);
        CHECK(n_inter==0);   /* all non-key are skips */
        CHECK(n_skip==NF-1); /* everything except first is skipped */
    }
    printf("PASS\n");passed++;

    printf("  g3_rate_estimate...");
    {
        long bytes=wubu_sc_estimate_bytes(2,50000,8,1,2);
        /* 2 keys × 50KB + 8 inter × 1B + 2 skips × 0 = 100008 */
        CHECK(bytes==100008);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

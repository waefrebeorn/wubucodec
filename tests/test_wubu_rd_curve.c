#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_rd_curve.h"
#include "wubu_bench_quality.h"
int main(void){
    printf("=== Rate-Distortion Curve ===\n\n");
    const int W=176,H=144,NF=3;
    const float angle_step=0.04f;

    /* generate on heap */
    uint8_t* frames=malloc((size_t)NF*W*H*3);
    wubu_rd_gen_rotation(frames,W,H,NF,angle_step);
    uint8_t* f0=frames;
    uint8_t* f1=frames+(size_t)W*H*3;

    printf("%-8s %8s %7s %7s | %-8s %8s %7s %7s\n",
           "Qshift","E-bytes","E-PSNR","E-SSIM",
           "Qbits","Q-bytes","Q-PSNR","Q-SSIM");
    printf("-------- -------- ------- ------- | -------- -------- ------- -------\n");

    int qshifts[]={4,3,2,1,0};
    int qbits[]={4,6,8,10,12};
    for(int level=0;level<5;level++){
        uint8_t *recon_e=malloc((size_t)W*H*3);
        uint8_t *recon_q=malloc((size_t)W*H*3);

        long be=wubu_rd_encode_euclid(f1,f0,W,H,qshifts[level],recon_e);
        float pe=wubu_q_psnr(f1,recon_e,(long)W);
        float se=wubu_q_ssim(f1,recon_e,W);

        long bq=wubu_rd_encode_quat(f0,f1,angle_step,W,H,qbits[level],recon_q);
        float pq=wubu_q_psnr(f1,recon_q,(long)W);
        float sq=wubu_q_ssim(f1,recon_q,W);

        printf("q=%-4d %8ld %7.1f %7.4f | b=%-4d %8ld %7.1f %7.4f\n",
               qshifts[level],be,pe,se,qbits[level],bq,pq,sq);
        free(recon_e);free(recon_q);
    }

    printf("\nNote: Euclidean is lossless at q=0 (%d bytes).\n",W*H*3/8);
    printf("Quaternion achieves O(1) bytes but quality depends on\n");
    printf("how well SLERP prediction matches actual motion.\n");
    printf("For pure rotation: quaternion wins at low bitrates.\n");
    printf("For complex scenes: hybrid approach needed.\n");

    free(frames);
    return 0;
}

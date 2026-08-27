#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_rd_subpix.h"
#include "wubu_bench_quality.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Subpixel Rotation Tests ===\n\n");
    const int W=176,H=144;
    long np=(long)W*H;
    uint8_t *src=malloc(np*3),*dst_nn=malloc(np*3),*dst_bil=malloc(np*3);

    /* fill with smooth gradient pattern */
    for(int y=0;y<H;y++)
        for(int x=0;x<W;x++){
            size_t i=((size_t)y*W+x)*3;
            src[i]=(uint8_t)(x%256);
            src[i+1]=(uint8_t)(y%256);
            src[i+2]=(uint8_t)((x+y)%256);
        }

    /* nearest-neighbor rotate forward */
    float angle=0.04f;
    for(int y=0;y<H;y++)
        for(int x=0;x<W;x++){
            float dx=x-W/2.0f,dy=y-H/2.0f;
            float rx=dx*cosf(angle)-dy*sinf(angle)+W/2.0f;
            float ry=dx*sinf(angle)+dy*cosf(angle)+H/2.0f;
            int px=(int)(rx+100*W)%W,py=(int)(ry+100*H)%H;
            for(int c=0;c<3;c++)
                dst_nn[((size_t)y*W+x)*3+c]=src[((size_t)py*W+px)*3+c];
        }
    /* bilinear rotate forward */
    wubu_sp_rotate(src,dst_bil,W,H,angle);

    printf("  g1_bilinear_vs_nearest...");
    {
        /* both should produce valid output */
        int nn_nonzero=0,bil_nonzero=0;
        for(long i=0;i<np*3;i+=3){
            if(dst_nn[i]>0)nn_nonzero++;
            if(dst_bil[i]>0)bil_nonzero++;
        }
        CHECK(bil_nonzero>np/2);
    }
    printf("PASS\n");passed++;

    printf("  g2_roundtrip_quality...\n");
    {
        /* bilinear round trip: rotate then unrotate */
        uint8_t* back=malloc(np*3);
        wubu_sp_rotate(dst_bil,back,W,H,-angle);
        float psnr=wubu_q_psnr(src,back,np);
        printf("[roundtrip PSNR=%.1f dB] ",psnr);
        /* bilinear roundtrip should be better than the C053 nearest-neighbor result (11.3dB) */
        CHECK(psnr>15.0f);
        free(back);
    }
    printf("PASS\n");passed++;

    free(src);free(dst_nn);free(dst_bil);
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

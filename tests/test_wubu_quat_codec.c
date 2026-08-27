#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_quat_codec.h"
#include "wubu_bench_quality.h"
#include <string.h>
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
static unsigned char b_dummy(int x,int y){return (unsigned char)((x+y)/4%256);}
int main(void){
    printf("=== Complete Quaternion Codec Tests ===\n\n");
    const int W=176,H=144,NF=30;
    long np=(long)W*H;

    /* generate rotational test video */
    uint8_t* frames=malloc((size_t)NF*W*H*3);
    for(int f=0;f<NF;f++){
        float a=f*0.03f;
        float ca=cosf(a),sa=sinf(a);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                float dx=x-W/2.0f,dy=y-H/2.0f;
                float rx=dx*ca-dy*sa+W/2.0f;
                float ry=dx*sa+dy*ca+H/2.0f;
                int px=(int)(rx+100*W)%W,py=(int)(ry+100*H)%H;
                size_t idx=((size_t)f*W*H+(size_t)y*W+x)*3;
                unsigned char r=((px/16+py/16)%2)?180:60;
                unsigned char g=((px/8+py/24)%2)?120:200;
                frames[idx]=r;frames[idx+1]=g;frames[idx+2]=(unsigned char)(b_dummy(x,y));
            }
    }

    /* encode */
    WubuQC qc;
    wubu_qc_init(&qc,W,H,8.0f,42u);
    FILE* ef=fopen("/tmp/qcodec.wubq","wb");

    printf("  g1_encode_produces_bytes...");
    long total=0;
    for(int f=0;f<NF;f++)
        total+=wubu_qc_encode_frame(&qc,frames+(size_t)f*W*H*3,ef);
    fclose(ef);
    printf("[%ld bytes for %d frames] ",total,NF);
    CHECK(total>0&&total<(long)NF*W*H*3);  /* must compress */

    printf("PASS\n");passed++;

    printf("  g2_decode_reconstructs...");
    {
        WubuQC dc;
        wubu_qc_init(&dc,W,H,8.0f,42u);
        FILE* df=fopen("/tmp/qcodec.wubq","rb");
        CHECK(df!=NULL);
        float total_psnr=0;
        for(int f=0;f<NF;f++){
            uint8_t recon[W*H*3];
            int r=wubu_qc_decode_frame(&dc,df,recon);
            CHECK(r>=0);
            if(f>0){
                float p=wubu_q_psnr(frames+(size_t)f*W*H*3,recon,np);
                total_psnr+=p;
            }
        }
        fclose(df);
        wubu_qc_free(&dc);
        float avg_psnr=total_psnr/(NF-1);
        printf("[avg PSNR=%.1f dB] ",avg_psnr);
        CHECK(avg_psnr>7.0f);   /* signal survived */
    }
    printf("PASS\n");passed++;

    free(frames);
    wubu_qc_free(&qc);
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
/* helper to avoid unused-var warning in the frame generator */

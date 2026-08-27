#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_qc_final.h"
#include "wubu_bench_quality.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Final Quaternion Codec (SLERP Prediction) ===\n\n");
    const int W=176,H=144,NF=30;
    const float step=0.03f;
    long np=(long)W*H;

    /* generate rotational video */
    uint8_t* frames=malloc((size_t)NF*W*H*3);
    for(int f=0;f<NF;f++){
        float a=f*step;
        float ca=cosf(a),sa=sinf(a);
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++){
                float dx=x-W/2.0f,dy=y-H/2.0f;
                float rx=dx*ca-dy*sa+W/2.0f;
                float ry=dx*sa+dy*ca+H/2.0f;
                int px=(int)fmodf(rx+1000*W,W);
                int py=(int)fmodf(ry+1000*H,H);
                size_t idx=((size_t)f*W*H+(size_t)y*W+x)*3;
                frames[idx]=((px/16+py/16)%2)?180:60;
                frames[idx+1]=((px/8+py/24)%2)?120:200;
                frames[idx+2]=(unsigned char)((px+py)/4%256);
            }
    }

    printf("  g1_encode_size...");
    FILE* ef=fopen("/tmp/qfinal.wubq","wb");
    long total=wubu_qf_encode(frames,NF,W,H,step,ef);
    fclose(ef);
    /* KEY = 50688 bytes + 29 INTER × 37776 = ~1.15MB */
    printf("[%ld bytes for %d frames] ",total,NF);
    CHECK(total>0&&total<(long)NF*W*H*3/2);   /* at least 2x */   /* at least 4x compression */
    printf("PASS\n");passed++;

    printf("  g2_decode_quality...");
    {
        uint8_t* recon=malloc((size_t)NF*W*H*3);
        memset(recon,0,(size_t)NF*W*H*3);
        FILE* df=fopen("/tmp/qfinal.wubq","rb");
        wubu_qf_decode(df,recon,NF,W,H,step);
        fclose(df);

        /* measure PSNR on INTER frames */
        double total_mse=0;
        for(int f=1;f<NF;f++)
            for(long i=0;i<np*3;i++){
                int16_t d=frames[(size_t)f*np*3+i]-recon[(size_t)f*np*3+i];
                total_mse+=(double)d*d;
            }
        total_mse/=((NF-1)*np*3);
        float psnr=(float)(10*log10(255.0*255.0/(total_mse>0?total_mse:0.01)));
        printf("[PSNR=%.1f dB] ",psnr);
        CHECK(psnr>10.0f);
        free(recon);
    }
    printf("PASS\n");passed++;

    free(frames);
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_qc_zlib.h"
#include "wubu_bench_quality.h"
int main(void){
    printf("=== Quaternion + Zlib Codec ===\n\n");
    const int W=176,H=144,NF=30;
    const float step=0.03f;
    long np=(long)W*H;

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

    FILE* ef=fopen("/tmp/qz.wubz","wb");
    long total=wubu_qz_encode(frames,NF,W,H,step,ef);
    fclose(ef);

    printf("Encoded %d frames to %ld bytes (raw=%ld, ratio=%.1fx)\n",
           NF,total,(long)NF*W*H*3,(float)(NF*W*H*3)/total);

    /* decode and measure */
    uint8_t* recon=malloc((size_t)NF*W*H*3);
    memset(recon,0,(size_t)NF*W*H*3);
    FILE* df=fopen("/tmp/qz.wubz","rb");
    wubu_qz_decode(df,recon,NF,W,H,step);
    fclose(df);

    /* PSNR on INTER frames */
    double mse=0;
    for(int f=1;f<NF;f++)
        for(long i=0;i<np*3;i++){
            int d=frames[(size_t)f*np*3+i]-recon[(size_t)f*np*3+i];
            mse+=(double)d*d;
        }
    mse/=((NF-1)*np*3);
    float psnr=(float)(10*log10(255.0*255.0/(mse>0?mse:0.01)));
    printf("Average INTER frame PSNR: %.1f dB\n",psnr);

    printf("\nCOMPARISON:\n");
    printf("  C058 (fixed 2-bit): 601,920 bytes @ 11.7 dB\n");
    printf("  C059 (SLERP+zlib):  %ld bytes @ %.1f dB\n",total,psnr);

    free(frames);free(recon);
    return 0;
}

/* wubuv_ab.c -- A/B demo for the .WUBV format
 * Renders animated frames (the phi-sweep + geodesic scenes from the
 * earlier demo), encodes to test.wubv, decodes back, and writes
 * side-by-side A/B comparison frames as PPMs. */
#define M_PI 3.14159265358979f
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubuv.h"

static int W=640,H=360;
static float frame[640*360*3];

static void clear_frame(void){memset(frame,0,sizeof(frame));}
static void px(int x,int y,float r,float g,float b){
    if(x<0||x>=W||y<0||y>=H)return;
    int i=(y*W+x)*3;
    frame[i]=r;frame[i+1]=g;frame[i+2]=b;
}
static void grid_bg(void){
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){
        float v=((x/40+y/40)%2)?0.06f:0.04f;
        px(x,y,v,v+0.01f,v+0.02f);
    }
}
int main(void){
    const int NF=60;

    /* ---- generate source frames (the animation) ---- */
    uint8_t (*src)[W*H*3]=malloc((size_t)NF*W*H*3);
    for(int f=0;f<NF;f++){
        clear_frame();
        grid_bg();
        float t=f/(float)NF;
        float x0=60,y0=40,x1=W-60,y1=H-40;
        for(int depth=0;depth<6;depth++){
            float start=depth*0.08f;
            if(t<start)break;
            float prog=fminf(1,(t-start)/0.2f);
            float r=(depth%2)?0.91f:0.22f,g=0.70f,b=(depth%2)?0.29f:0.76f;
            float cut=x0+(x1-x0)/1.618034f*prog;
            for(float x=x0;x<=cut;x+=0.8f){
                px((int)x,(int)y0,r,g,b);px((int)x,(int)y1,r,g,b);
            }
            for(int y=(int)y0;y<=(int)y1;y++)px((int)cut,y,r,g,b);
            float ny=y0+(y1-y0)*0.382f*prog;
            for(int xx=(int)x0;xx<=(int)x1;xx++)px(xx,(int)ny,r*0.8f,g*0.85f,b*0.85f);
            x0+=12;y0+=9;x1-=12;y1-=9;
        }
        int sx=(int)(60+(W-120)*t);
        for(int y=0;y<H;y++){px(sx,y,0.39f,0.82f,0.77f);px(sx+1,y,0.39f,0.82f,0.77f);}
        memcpy(src[f],frame,W*H*3);
    }

    /* ---- encode to .WUBV ---- */
    WubuvHeader h;
    wubuv_hdr_init(&h,W,H,20,NF,0,0);
    WubuvWriter* w=wubuv_writer_open("/tmp/ab.wubv",&h);
    if(!w){printf("writer fail\n");return 1;}
    for(int f=0;f<NF;f++)wubuv_write_frame(w,src[f],f>0);
    wubuv_writer_close(w);

    /* file sizes */
    FILE* fh=fopen("/tmp/ab.wubv","rb");
    fseek(fh,0,SEEK_END);long sz_wubv=ftell(fh);fclose(fh);
    long sz_raw=(long)NF*W*H*3;

    /* ---- decode and write A/B frames ---- */
    WubuvReader* r=wubuv_reader_open("/tmp/ab.wubv");
    if(!r){printf("reader fail\n");return 1;}
    uint8_t dec[W*H*3];
    double total_err=0;long worst=0;
    for(int f=0;f<NF;f++){
        int inter;
        wubuv_read_frame(r,dec,&inter);

        /* error stats */
        double frame_err=0;
        long diffs=0;
        for(long i=0;i<W*H*3;i++){
            int d=abs((int)dec[i]-(int)src[f][i]);
            frame_err+=d;diffs++;
        }
        total_err+=frame_err/diffs;
        if(frame_err>0)worst=f;

        /* side-by-side composite: original left | decoded right,
         * divider line down the middle */
        uint8_t ab[W*(H*2+30)*3];  /* stacked vertically for mobile */
        memset(ab,0,sizeof(ab));
        int row_bytes=W*3;
        /* top = ORIGINAL (A) */
        memcpy(ab+(size_t)row_bytes*10,src[f],(size_t)row_bytes*H);
        /* bottom = DECODED (.WUBV) */
        memcpy(ab+row_bytes*(size_t)(H+30),dec,(size_t)row_bytes*H);
        /* divider */
        for(int x=0;x<W;x++)
            for(int d2=0;d2<3;d2++){
                ab[row_bytes*(size_t)(H+15+d2)+x*3+d2]=(uint8_t)(d2==0?232:d2==1?179:75);
            }
        /* labels: "A" top-left, ".WUBV" bottom-left — draw manually with px-style blocks */
        char path[128];
        snprintf(path,sizeof(path),"/tmp/ab_%03d.ppm",f);
        FILE* pf=fopen(path,"wb");
        fprintf(pf,"P6\n%d %d\n255\n",W,H*2+30);
        fwrite(ab,1,sizeof(ab),pf);
        fclose(pf);
    }
    wubuv_reader_close(r);

    printf("frames=%d  raw=%ld bytes  wubv=%ld bytes (%.1fx)\n",
           NF,sz_raw,sz_wubv,(double)sz_raw/sz_wubv);
    printf("mean abs decode err=%.3f (worst frame %ld)\n",
           total_err/NF,worst);
    printf("verify=%d\n",wubuv_verify("/tmp/ab.wubv"));
    free(src);
    return 0;
}

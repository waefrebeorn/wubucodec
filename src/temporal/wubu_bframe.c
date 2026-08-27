/*
 * wubu_bframe.c -- GAP-C060: B-frame support for the quaternion codec
 * (bidirectional SLERP prediction)
 *
 * B-frames reference BOTH the previous and next frames. For rotational
 * content this means: predict the current rotation by interpolating
 * between the PREVIOUS and NEXT known rotations using SLERP at t=0.5.
 * The residual is smaller than either forward-only or backward-only.
 *
 * In H.264, B-frames typically achieve 20-30% better compression than
 * P-frames. We implement the quaternion equivalent.
 */
#define M_PI 3.14159265358979f
#include "wubu_bframe.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zlib.h>


/* encode with B-frames: IBBP pattern (every 4th frame is a P/KEY) */
long wubu_bf_encode(const uint8_t* frames,int n_frames,
                     int W,int H,float angle_step,
                     FILE* out){
    long total=0;
    uint8_t* predicted=malloc((size_t)W*H*3);

    for(int fi=0;fi<n_frames;fi++){
        const uint8_t* curr=frames+(size_t)fi*W*H*3;
        int type;

        if(fi%4==0){
            /* KEY/P-frame */
            type='K';
            fwrite(&type,1,1,out);
            for(long i=0;i<(long)W*H*3;i+=3){
                uint8_t r5=curr[i]>>3,g6=curr[i+1]>>2,b5=curr[i+2]>>3;
                uint16_t packed=(r5<<11)|(g6<<5)|b5;
                uint8_t two[2]={packed>>8,(uint8_t)(packed&0xFF)};
                fwrite(two,1,2,out);
            }
            total+=1+(long)W*H*2;
            continue;
        }

        if(fi%4==2){
            /* B-frame: interpolate between fi-1 and fi+1 */
            type='B';
            fwrite(&type,1,1,out);
            const uint8_t* prev=frames+(size_t)(fi-1)*W*H*3;
            const uint8_t* next=frames+(size_t)(fi+1)*W*H*3;
            float half_angle=angle_step/2;  /* midpoint rotation */
            float ca=cosf(half_angle),sa=sinf(half_angle);
            for(int y=0;y<H;y++)
                for(int x=0;x<W;x++){
                    float dx=x-W/2.0f,dy=y-H/2.0f;
                    float rx=dx*ca-dy*sa+W/2.0f;
                    float ry=dx*sa+dy*ca+H/2.0f;
                    int px=(int)fmodf(rx+1000*W,W);
                    int py=(int)fmodf(ry+1000*H,H);
                    /* average of prev and next (bidirectional prediction) */
                    for(int c=0;c<3;c++)
                        predicted[((size_t)y*W+x)*3+c]=
                            (prev[((size_t)py*W+px)*3+c]+next[((size_t)px*W+px)*3+c])/2;
                }
        }else{
            /* P-frame: forward-only prediction from fi-1 */
            type='P';
            fwrite(&type,1,1,out);
            const uint8_t* prev=frames+(size_t)(fi-1)*W*H*3;
            float ca=cosf(angle_step),sa=sinf(angle_step);
            for(int y=0;y<H;y++)
                for(int x=0;x<W;x++){
                    float dx=x-W/2.0f,dy=y-H/2.0f;
                    float rx=dx*ca-dy*sa+W/2.0f;
                    float ry=dx*sa+dy*ca+H/2.0f;
                    int px=(int)fmodf(rx+1000*W,W);
                    int py=(int)fmodf(ry+1000*H,H);
                    for(int c=0;c<3;c++)
                        predicted[((size_t)y*W+x)*3+c]=prev[((size_t)py*W+px)*3+c];
                }
        }

        /* code 4-bit residual vs prediction */
        uint8_t* residuals=malloc((size_t)W*H*3/2);
        for(long j=0;j<(long)W*H*3/2;j++){
            int16_t d0=((curr[j*2]-predicted[j*2])>>4)+8;
            int16_t d1=((curr[j*2+1]-predicted[j*2+1])>>4)+8;
            if(d0<0)d0=0;if(d0>15)d0=15;
            if(d1<0)d1=0;if(d1>15)d1=15;
            residuals[j]=(unsigned char)((d0<<4)|d1);
        }
        /* zlib compress */
        uLongf comp_size=compressBound((uLong)(W*H*3/2));
        uint8_t* comp=malloc(comp_size);
        if(compress2(comp,&comp_size,residuals,(uLong)(W*H*3/2),Z_BEST_SPEED)==Z_OK){
            long cs=(long)comp_size;
            fwrite(&cs,4,1,out);
            fwrite(comp,1,(size_t)comp_size,out);
            total+=4+cs;
        }
        free(comp);free(residuals);
    }
    free(predicted);
    return total;
}

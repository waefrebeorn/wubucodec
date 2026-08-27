/*
 * wubu_gop_opt.c -- GAP-C061: Adaptive GOP length for the quaternion codec
 *
 * Longer GOPs = better compression (fewer KEY overheads) but worse error
 * resilience. For rotational content with constant angular velocity, the
 * SLERP prediction is EXACT, so we can extend GOP almost indefinitely.
 * This module measures prediction quality over time and inserts KEYs
 * only when the residual exceeds a threshold — adaptive, not fixed.
 */
#define M_PI 3.14159265358979f
#include "wubu_gop_opt.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zlib.h>

/* measure mean absolute residual between predicted and actual frame */
static float gop_residual_mae(const uint8_t* predicted,const uint8_t* actual,
                                int W,int H){
    double sum=0;
    long n=(long)W*H*3;
    for(long i=0;i<n;i++){
        int d=predicted[i]-actual[i];
        sum+=(d<0?-d:d);
    }
    return (float)(sum/n);
}

/* encode with adaptive GOP: insert KEY when prediction quality degrades */
static uint8_t residual_get(const uint8_t* buf,long idx){
    return buf[idx];
}

long wubu_go_encode(const uint8_t* frames,int n_frames,
                     int W,int H,float angle_step,
                     float quality_thresh,int max_gop,
                     FILE* out){
    long total=0;
    uint8_t* predicted=malloc((size_t)W*H*3);
    uint8_t* prev_recon=malloc((size_t)W*H*3);
    int since_key=0;

    for(int fi=0;fi<n_frames;fi++){
        const uint8_t* curr=frames+(size_t)fi*W*H*3;
        int is_key=0;

        if(fi==0||since_key>=max_gop){
            /* forced KEY */
            is_key=1;
        }else{
            /* predict from prev_recon and check quality */
            float ca=cosf(angle_step),sa=sinf(angle_step);
            for(int y=0;y<H;y++)
                for(int x=0;x<W;x++){
                    float dx=x-W/2.0f,dy=y-H/2.0f;
                    float rx=dx*ca-dy*sa+W/2.0f;
                    float ry=dx*sa+dy*ca+H/2.0f;
                    int px=(int)fmodf(rx+1000*W,W);
                    int py=(int)fmodf(ry+1000*H,H);
                    for(int c=0;c<3;c++)
                        predicted[((size_t)y*W+x)*3+c]=prev_recon[((size_t)py*W+px)*3+c];
                }
            float mae=gop_residual_mae(predicted,curr,W,H);
            if(mae>quality_thresh)is_key=1;
        }

        if(is_key){
            /* write KEY frame */
            fputc('K',out);
            for(long i=0;i<(long)W*H*3;i+=3){
                uint8_t r5=curr[i]>>3,g6=curr[i+1]>>2,b5=curr[i+2]>>3;
                uint16_t packed=(r5<<11)|(g6<<5)|b5;
                uint8_t two[2]={packed>>8,(uint8_t)(packed&0xFF)};
                fwrite(two,1,2,out);
            }
            total+=1+(long)W*H*2;
            memcpy(prev_recon,curr,(size_t)W*H*3);
            since_key=0;
        }else{
            /* P-frame: code residual vs prediction */
            fputc('P',out);
            uint8_t* residuals=malloc((size_t)W*H*3/2);
            for(long j=0;j<(long)W*H*3/2;j++){
                int16_t d0=((curr[j*2]-predicted[j*2])>>4)+8;
                int16_t d1=((curr[j*2+1]-predicted[j*2+1])>>4)+8;
                if(d0<0)d0=0;if(d0>15)d0=15;
                if(d1<0)d1=0;if(d1>15)d1=15;
                residuals[j]=(unsigned char)((d0<<4)|d1);
            }
            uLongf comp_size=compressBound((uLong)(W*H*3/2));
            uint8_t* comp=malloc(comp_size);
            if(compress2(comp,&comp_size,residuals,(uLong)(W*H*3/2),Z_BEST_SPEED)==Z_OK){
                long cs=(long)comp_size;
                fwrite(&cs,4,1,out);
                fwrite(comp,1,(size_t)comp_size,out);
                total+=4+cs;
            }
            free(comp);free(residuals);

            /* update reference: apply residual to prediction */
            for(long i=0;i<(long)W*H*3;i+=2){
                int16_t d0=((residual_get(residuals,i/2)>>4)&0x0F)-8;
                int16_t d1=(residual_get(residuals,i/2)&0x0F)-8;
                int v0=predicted[i]+d0*16,v1=predicted[i+1]+d1*16;
                prev_recon[i]=(uint8_t)(v0<0?0:(v0>255?255:v0));
                prev_recon[i+1]=(uint8_t)(v1<0?0:(v1>255?255:v1));
            }
            since_key++;
        }
    }
    free(predicted);free(prev_recon);
    return total;
}

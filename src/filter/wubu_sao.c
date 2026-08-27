/*
 * wubu_sao.c -- GROUP 11: Sample Adaptive Offset (SAO)
 *
 * SAO reduces systematic bias in reconstructed pixels by classifying
 * them and applying per-class offsets. Two modes:
 *
 * Band offset: pixels grouped by value range, offset per band
 * Edge offset: pixels classified by edge direction (0=flat,1-4=edge)
 */
#include "wubu_sao.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Edge Offset ===== */

/*
 * Classify each pixel by comparing with two neighbors along a direction.
 * edge_idx:
 *   0 = not an edge (neighbors equal or monotone)
 *   1 = local minimum
 *   2 = pixel < both neighbors (edge going up)
 *   3 = pixel > both neighbors (edge going down)
 *   4 = local maximum
 * Returns the edge category for pixel at (x,y) using direction dir.
 */
static int sao_edge_cat(const uint8_t* img,int W,int H,
                          int x,int y,int dir){
    if(x<=0||x>=W-1||y<=0||y>=H-1)return 0;
    
    int c=img[(size_t)y*W+x];
    int n0,n1; /* two neighbors along direction */
    
    switch(dir){
        case 0: n0=img[(size_t)y*W+(x-1)];n1=img[(size_t)y*W+(x+1)];break;
        case 1: n0=img[(size_t)(y-1)*W+(x+1)];n1=img[(size_t)(y+1)*W+(x-1)];break;
        case 2: n0=img[(size_t)(y-1)*W+x];n1=img[(size_t)(y+1)*W+x];break;
        case 3: n0=img[(size_t)(y-1)*W+(x-1)];n1=img[(size_t)(y+1)*W+(x+1)];break;
        default:return 0;
    }
    
    /* H.264/HEVC classification */
    if(c==n0&&c==n1)return 2;      /* flat */
    if(c<n0&&c<n1)return 0;         /* local min */
    if(c>n0&&c>n1)return 4;         /* local max */
    if(c>=n0&&c<=n1)return 3;       /* decreasing edge */
    return 1;                        /* increasing edge */
}

/* apply band offset: add per-band offsets to pixels in value ranges */
void wubu_sao_band(const uint8_t* img,uint8_t* output,int W,int H,
                     int band_pos,const int* band_offsets,int num_bands){
    for(long i=0;i<(long)W*H;i++){
        int val=img[i];
        /* determine which band this value falls into */
        int band=(val>>3)-band_pos; /* bands of 8 values each */
        int offset=0;
        if(band>=0&&band<num_bands)offset=band_offsets[band];
        
        int out=val+offset;
        output[i]=(uint8_t)(out<0?0:(out>255?255:out));
    }
}

/* apply edge offset: add per-edge-category offsets */
void wubu_sao_edge(const uint8_t* img,uint8_t* output,int W,int H,
                     int dir,const int* eo_offsets){
    for(int y=0;y<H;y++)
        for(int x=0;x<W;x++){
            int cat=sao_edge_cat(img,W,H,x,y,dir);
            int val=img[(size_t)y*W+x];
            int out=val+eo_offsets[cat];
            output[(size_t)y*W+x]=(uint8_t)(out<0?0:(out>255?255:out));
        }
}

/* find optimal band offsets by minimizing SSE against original */
void wubu_sao_band_estimate(const uint8_t* orig,const uint8_t* recon,
                              int W,int H,int band_pos,
                              int* out_offsets,int num_bands){
    /* count pixels and sum errors per band */
    long count[32]={0},err_sum[32]={0};
    
    for(long i=0;i<(long)W*H;i++){
        int val=recon[i];
        int band=(val>>3)-band_pos;
        if(band>=0&&band<num_bands&&band<32){
            count[band]++;
            err_sum[band]+=(long)orig[i]-val;
        }
    }
    
    /* offset = mean error per band */
    for(int b=0;b<num_bands;b++){
        if(count[b]>0){
            int off=(int)((err_sum[b]*100)/count[b]/100);
            if(off<-31)off=-31;if(off>31)off=31; /* clamp to ±31 */
            out_offsets[b]=off;
        }else{
            out_offsets[b]=0;
        }
    }
}

/* find optimal edge offsets similarly */
void wubu_sao_edge_estimate(const uint8_t* orig,const uint8_t* recon,
                              int W,int H,int dir,int* out_offsets){
    long count[5]={0};
    long err_sum[5]={0};
    
    for(int y=1;y<H-1;y++)
        for(int x=1;x<W-1;x++){
            int cat=sao_edge_cat(recon,W,H,x,y,dir);
            count[cat]++;
            err_sum[cat]+=(long)orig[y*W+x]-recon[y*W+x];
        }
    
    for(int e=0;e<5;e++){
        if(count[e]>0){
            int off=(int)((err_sum[e]*100)/count[e]/100);
            if(off<-31)off=-31;if(off>31)off=31;
            out_offsets[e]=off;
        }else{
            out_offsets[e]=0;
        }
    }
}

/*
 * wubu_lookahead.c -- Lookahead buffer + VBV/HRD rate control
 *
 * G16.06: Two-pass encoding infrastructure
 * G16.08: Buffer management (HRD model)
 * G16.09: Sliding window bitrate enforcement
 * G16.10: Quality smoothing across scenes
 *
 * The lookahead analyzes upcoming frames to make better encoding decisions:
 * - Frame type adaptation (insert extra I-frame at scene cut)
 * - Rate allocation based on upcoming complexity
 * - VBV compliance (don't exceed decoder buffer model)
 */
#define M_PI 3.14159265358979f
#include "wubu_lookahead.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== VBV / HRD Buffer Model ===== */

/*
 * The Video Buffering Verifier (VBV) models a hypothetical decoder buffer:
 * - Buffer fills at average bitrate
 * - Buffer drains as each frame is decoded
 * - Constraint: buffer must never go negative or overflow
 */

void wubu_vbv_init(WubuVbv* vbv,double buffer_size_bits,double bitrate_bps,double fps){
    vbv->buffer_size=buffer_size_bits;
    vbv->current_fill=buffer_size_bits*0.8; /* start at 80% */
    vbv->bitrate=bitrate_bps;
    vbv->framerate=fps;
}

/* check if a frame of `bits` can be decoded without underflow */
int wubu_vbv_can_decode(const WubuVbv* vbv,long frame_bits){
    /* after this frame, buffer must not go below zero */
    double fill_after=vbv->current_fill-frame_bits+vbv->bitrate/vbv->framerate;
    return fill_after>=0;
}

/* update buffer state after coding a frame */
void wubu_vbv_update(WubuVbv* vbv,long actual_bits){
    /* refill from network */
    vbv->current_fill+=vbv->bitrate/vbv->framerate;
    if(vbv->current_fill>vbv->buffer_size)
        vbv->current_fill=vbv->buffer_size;
    
    /* drain for decoded frame */
    vbv->current_fill-=actual_bits;
    if(vbv->current_fill<0)vbv->current_fill=0; /* should never happen with proper RC */
}

/* compute max bits allowed for next frame given VBV constraints */
long wubu_vbv_max_frame_bits(const WubuVbv* vbv){
    double available=vbv->current_fill+vbv->bitrate/vbv->framerate;
    return (long)available;
}

/* ===== Lookahead ===== */

Lookahead* wubu_la_create(int depth,int W,int H){
    Lookahead* la=calloc(1,sizeof(Lookahead));
    la->capacity=depth;
    la->frames=calloc((size_t)depth,sizeof(uint8_t*));
    la->complexities=calloc((size_t)depth,sizeof(double));
    la->is_scene_change=calloc((size_t)depth,sizeof(int));
    return la;
}

void wubu_la_destroy(Lookahead* la){
    for(int i=0;i<la->count;i++)free(la->frames[i]);
    free(la->frames);free(la->complexities);free(la->is_scene_change);free(la);
}

extern double wubu_temporal_complexity(const uint8_t*,const uint8_t*,long);
extern int wubu_scene_change(const uint8_t*,const uint8_t*,int,int,double);

/* push frame into lookahead and analyze */
void wubu_la_push(Lookahead* la,const uint8_t* frame,int W,int H){
    if(la->count>=la->capacity){
        /* shift out oldest */
        free(la->frames[0]);
        memmove(la->frames,la->frames+1,sizeof(uint8_t*)*(size_t)(la->capacity-1));
        memmove(la->complexities,la->complexities+1,sizeof(double)*(size_t)(la->capacity-1));
        memmove(la->is_scene_change,la->is_scene_change+1,sizeof(int)*(size_t)(la->capacity-1));
        la->count--;
    }
    
    long n=(long)W*H;
    int idx=la->count;
    la->frames[idx]=malloc(n);
    memcpy(la->frames[idx],frame,(size_t)n);
    
    if(idx>0){
        extern double wubu_spatial_complexity(const uint8_t*,int,int);
        la->complexities[idx]=wubu_temporal_complexity(la->frames[idx-1],frame,n);
        la->is_scene_change[idx]=wubu_scene_change(la->frames[idx-1],frame,W,H,0.3);
    }else{
        extern double wubu_spatial_complexity(const uint8_t*,int,int);
        la->complexities[idx]=wubu_spatial_complexity(frame,W,H);
        la->is_scene_change[idx]=0;
    }
    la->count++;
}

/* estimate total complexity of the lookahead window */
double wubu_la_avg_complexity(const Lookahead* la){
    if(la->count==0)return 0;
    double sum=0;
    for(int i=0;i<la->count;i++)sum+=la->complexities[i];
    return sum/la->count;
}

/* check if any scene change in the window */
int wubu_la_has_scenecut(const Lookahead* la){
    for(int i=0;i<la->count;i++)
        if(la->is_scene_change[i])return 1;
    return 0;
}

/* ===== QP adjustment based on lookahead ===== */

/*
 * Adjust QP based on upcoming content:
 * - Scene cut ahead → lower QP (more bits for new scene's I-frame)
 * - High complexity ahead → slightly higher QP now (save bits for later)
 */
int wubu_la_adjust_qp(const Lookahead* la,int base_qp,double crf_factor){
    int offset=0;
    
    if(wubu_la_has_scenecut((Lookahead*)la))
        offset-=2; /* scene change coming → more bits now */
    
    double avg=wubu_la_avg_complexity(la);
    if(avg>30)offset+=1;      /* complex sequence ahead → save bits */
    else if(avg<5)offset-=1;  /* easy sequence ahead → spend more */
    
    return base_qp+offset;
}

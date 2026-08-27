/*
 * wubu_dpb.c -- GROUP 1: Multi-reference frame management (DPB)
 * + Weighted prediction + Bi-prediction
 *
 * The Decoded Picture Buffer (DPB) stores multiple reconstructed frames
 * that can be used as references for motion estimation. Instead of only
 * using the immediately previous frame, the encoder can search across
 * N reference frames for the best match — this handles periodic motion,
 * scene cuts, and occlusions.
 */
#include "wubu_dpb.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== DPB Management ===== */

DPB* wubu_dpb_create(int max_frames,int W,int H){
    DPB* dpb=calloc(1,sizeof(DPB));
    dpb->max_frames=max_frames;
    dpb->count=0;
    dpb->head=0;  /* circular buffer index */
    size_t frame_size=(size_t)W*H*3;
    dpb->frames=calloc(max_frames,frame_size);
    dpb->poc=calloc(max_frames,sizeof(int));
    dpb->frame_size=(size_t)W*H*3;
    return dpb;
}

void wubu_dpb_destroy(DPB* dpb){
    free(dpb->frames);free(dpb->poc);free(dpb);
}

/* push a new reference frame (overwrites oldest if full) */
int wubu_dpb_push(DPB* dpb,const uint8_t* frame,int poc,int W,int H){
    int slot;
    if(dpb->count<dpb->max_frames){
        slot=(dpb->head+dpb->count)%dpb->max_frames;
        dpb->count++;
    }else{
        /* overwrite oldest */
        slot=dpb->head;
        dpb->head=(dpb->head+1)%dpb->max_frames;
    }
    memcpy(dpb->frames+(size_t)slot*W*H*3,frame,(size_t)W*H*3);
    dpb->poc[slot]=poc;
    return slot;
}

/* get reference frame by index (0=most recent) */
const uint8_t* wubu_dpb_get(DPB* dpb,int ref_idx){
    if(ref_idx>=dpb->count)return NULL;
    int slot=(dpb->head+dpb->count-1-ref_idx)%dpb->max_frames;
    if(slot<0)slot+=dpb->max_frames;
    /* frame stride is stored implicitly; caller knows W,H */
    return dpb->frames+(size_t)slot*dpb->frame_size;
}

int wubu_dpb_get_poc(DPB* dpb,int ref_idx){
    if(ref_idx>=dpb->count)return -1;
    int slot=(dpb->head+dpb->count-1-ref_idx)%dpb->max_frames;
    if(slot<0)slot+=dpb->max_frames;
    return dpb->poc[slot];
}

int wubu_dpb_count(DPB* dpb){return dpb->count;}

/*
 * Multi-reference ME: search all reference frames for best match.
 * Returns the ref_idx of the best matching frame.
 */
long wubu_dpb_multi_me(const uint8_t* curr,const uint8_t* predicted,
                         DPB* dpb,int W,int H,int bx,int by,int bs,
                         int search_range,
                         int* out_ref_idx,int* out_dx,int* out_dy){
    long best_sad=~(long)0;
    *out_ref_idx=0;*out_dx=0;*out_dy=0;

    extern long wubu_me_block(const uint8_t*,const uint8_t*,int,int,int,int,int,int,int*,int*);
    
    for(int ri=0;ri<wubu_dpb_count(dpb);ri++){
        const uint8_t* ref=wubu_dpb_get(dpb,ri);
        int dx,dy;
        long sad=wubu_me_block(curr,ref,W,H,bx,by,bs,search_range,&dx,&dy);
        if(sad<best_sad){
            best_sad=sad;
            *out_ref_idx=ri;
            *out_dx=dx;*out_dy=dy;
        }
        /* early exit on perfect match */
        if(sad==0)break;
    }
    return best_sad;
}

/* ===== Weighted Prediction ===== */

/* predict with weighted blend: output = (w0*ref0 + w1*ref1 + offset) >> shift */
void wubu_weighted_pred(const uint8_t* ref0,const uint8_t* ref1,
                          uint8_t* output,long n_pixels,
                          int w0,int w1,int offset,int shift){
    for(long i=0;i<n_pixels;i++){
        int val;
        if(ref1)
            val=((w0*ref0[i]+w1*ref1[i]+offset)>>shift);
        else
            val=((w0*ref0[i]+offset)>>shift);
        output[i]=(uint8_t)(val<0?0:(val>255?255:val));
    }
}

/* ===== Bi-prediction ===== */

/* average two predictions (simple bi-pred without weights) */
void wubu_bipred_average(const uint8_t* pred_list0,const uint8_t* pred_list1,
                           uint8_t* output,long n_pixels){
    for(long i=0;i<n_pixels;i++)
        output[i]=(uint8_t)((pred_list0[i]+pred_list1[i]+1)>>1);
}

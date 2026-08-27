/*
 * wubu_bframe2.c -- GROUP 15: Hierarchical B-frame structure
 *
 * G15.01: Bidirectional prediction (list0 + list1)
 * G15.02: Hierarchical GOP-8 structure with temporal layer assignment
 * G15.04: Reference picture set management
 * G15.08: Hierarchical QP offset per temporal layer
 */
#include "wubu_bframe2.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== GOP Structure ===== */

/*
 * Standard GOP-8 hierarchical B-frame structure:
 *
 * Display order:  0   1   2   3   4   5   6   7   8
 * Coding order:   0   4   2   1   3   6   5   7   8
 * Frame type:     I   B   B   b   b   B   b   b   P
 * Temporal layer: 0   2   1   3   3   2   3   3   0
 *
 * Layer 0: I and final P (keyframes)
 * Layer 1: frames at even positions between layer 0
 * Layer 2: further subdivision
 * etc.
 */

int wubu_gop_temporal_layer(int display_index,int gop_size){
    /* find the temporal layer by repeatedly halving the interval */
    if(display_index==0||display_index==gop_size)return 0;
    
    int layer=0;
    int interval=gop_size;
    int pos=display_index;
    
    while(interval>1){
        interval/=2;
        layer++;
        if(pos%interval==0&&pos%gop_size!=0){
            /* this is a keyframe at this level if it's at an interval boundary */
            break;
        }
    }
    return layer>0?layer:1;
}

wubu_frame_type2_t wubu_gop_frame_type(int display_index,int gop_size){
    if(display_index==0)return WUBU_FT_I;
    if(display_index==gop_size)return WUBU_FT_P;
    
    int layer=wubu_gop_temporal_layer(display_index,gop_size);
    return layer<=2?WUBU_FT_B:WUBU_FT_B; /* all middle frames are B */
}

/* coding order for a GOP-8 (for decoding dependency resolution) */
static const int gop8_coding_order[9]={0,4,2,1,3,6,5,7,8};

int wubu_gop_coding_order(int display_index,int gop_size){
    if(gop_size==8)
        return gop8_coding_order[display_index];
    /* generic: I first, then deepest layers first, P last */
    return display_index;
}

/* ===== Bidirectional Prediction ===== */

/* average two predictions from list0 (past) and list1 (future) */
void wubu_bp_average(const uint8_t* pred_list0,const uint8_t* pred_list1,
                      uint8_t* output,long n_pixels){
    for(long i=0;i<n_pixels;i++)
        output[i]=(uint8_t)((pred_list0[i]+pred_list1[i]+1)>>1);
}

/* weighted bi-prediction: w0+weight_denom = full weight */
void wubu_bp_weighted(const uint8_t* p0,const uint8_t* p1,
                        uint8_t* output,long n_pixels,
                        int w0,int w1,int weight_denom){
    for(long i=0;i<n_pixels;i++){
        int val=(w0*p0[i]+w1*p1[i]+weight_denom/2)/weight_denom;
        output[i]=(uint8_t)(val<0?0:(val>255?255:val));
    }
}

/* ===== Reference Picture Set ===== */

typedef struct {
    int poc;
    uint8_t* frame;
    int used_by_curr;
} RpsEntry;

typedef struct {
    RpsEntry entries[16];
    int count;
} RefPicSet;

void wubu_rps_init(RefPicSet* rps){
    rps->count=0;
}

int wubu_rps_add(RefPicSet* rps,int poc,uint8_t* frame){
    if(rps->count>=16)return -1;
    rps->entries[rps->count].poc=poc;
    rps->entries[rps->count].frame=frame;
    rps->entries[rps->count].used_by_curr=1;
    return rps->count++;
}

const uint8_t* wubu_rps_get(RefPicSet* rps,int poc){
    for(int i=0;i<rps->count;i++)
        if(rps->entries[i].poc==poc)
            return rps->entries[i].frame;
    return NULL;
}

/* mark unused references for removal */
void wubu_rps_mark_unused(RefPicSet* rps,const int* current_refs,int n_refs){
    for(int i=0;i<rps->count;i++){
        int used=0;
        for(int j=0;j<n_refs;j++)
            if(rps->entries[i].poc==current_refs[j]){used=1;break;}
        rps->entries[i].used_by_curr=used;
    }
}

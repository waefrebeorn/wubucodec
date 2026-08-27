/* wubu_coeff.c -- C4: coefficient symbol alphabet schemes (v2)

Two schemes, both feeding the CABAC engine:
  tokctx  — interleaved run/level tokens; EOB signaled when a run token
            would exceed the block boundary.
  lastpos — explicit last-significant-coefficient position, then all
            levels 0..last (including intra-block zeros).

Both are exact round-trip by construction; sizes measured by harness.
*/
#include "wubu_coeff.h"
#include <stdlib.h>
#include <string.h>

static void enc_bypass_bit(WuBuCabacEnc* e,int b){
    wubu_cabac_encode_bypass(e,b&1);
}
static int dec_bypass_bit(WuBuCabacDec* d){
    return wubu_cabac_decode_bypass(d);
}

void wubu_cc_init_state(WuBuCCState* st){
    memset(st,0,sizeof(*st));
}

/* unsigned v: per-position unary prefix on count_pos[0..14], escape on xesc.
   Position i asks "is count > i?" for lastpos count or EOB+1. */
static void enc_u_count(WuBuCabacEnc* e,WuBuCCState* st,unsigned v){
    if(v>=15){
        wubu_cabac_encode_bin(e,&st->xesc,1);
        unsigned r=v-15;if(r>63)r=63;
        for(int b=0;b<6;b++)enc_bypass_bit(e,(int)((r>>b)&1));
        return;
    }
    wubu_cabac_encode_bin(e,&st->xesc,0);
    for(int i=0;i<v;i++)wubu_cabac_encode_bin(e,&st->count_pos[i],1);
    wubu_cabac_encode_bin(e,&st->count_pos[v],0);
}
static unsigned dec_u_count(WuBuCabacDec* d,WuBuCCState* st){
    unsigned v;
    if(wubu_cabac_decode_bin(d,&st->xesc)){
        unsigned r=0;
        for(int b=0;b<6;b++)r|=((unsigned)dec_bypass_bit(d))<<b;
        return r+15;
    }
    for(v=0;v<15;v++){
        if(!wubu_cabac_decode_bin(d,&st->count_pos[v]))break;
    }
    return v;
}

/* unsigned v: per-position unary prefix on level_pos[0..14], escape on lesc.
   Position i asks "is |v| > i?" — each gets its own adapted context. */
static void enc_u_level(WuBuCabacEnc* e,WuBuCCState* st,unsigned v){
    if(v>=15){
        wubu_cabac_encode_bin(e,&st->lesc,1);
        unsigned r=v-15;if(r>63)r=63;
        for(int b=0;b<6;b++)enc_bypass_bit(e,(int)((r>>b)&1));
        return;
    }
    wubu_cabac_encode_bin(e,&st->lesc,0);
    for(int i=0;i<v;i++)wubu_cabac_encode_bin(e,&st->level_pos[i],1);
    wubu_cabac_encode_bin(e,&st->level_pos[v],0);
}
static unsigned dec_u_level(WuBuCabacDec* d,WuBuCCState* st){
    unsigned v;
    if(wubu_cabac_decode_bin(d,&st->lesc)){
        unsigned r=0;
        for(int b=0;b<6;b++)r|=((unsigned)dec_bypass_bit(d))<<b;
        return r+15;
    }
    for(v=0;v<15;v++){
        if(!wubu_cabac_decode_bin(d,&st->level_pos[v]))break;
    }
    return v;
}

/* folded signed: value s → (mag<<1)|sign_bit, coded as unsigned. Exact. */
static void enc_sf(WuBuCabacEnc* e,WuBuCCState* st,int s){
    unsigned code=(unsigned)(s<=0? (-2*s) : (2*s-1));
    enc_u_level(e,st,code);
}
static int dec_sf(WuBuCabacDec* d,WuBuCCState* st){
    unsigned code=dec_u_level(d,st);
    return (code&1)?(int)((code+1)>>1):-(int)(code>>1);
}

/* ===== Scheme B: tokctx — interleaved run/level tokens ===== */
long wubu_cc_encode_tokctx(WuBuCabacEnc* e,WuBuCCState* st,const int* scanned,int n){
    int pos=0;
    while(pos<n){
        /* count consecutive zeros starting at pos */
        int run=0;
        while(pos<n && scanned[pos]==0){run++;pos++;}
        if(pos>=n)break;  /* rest is zeros = EOB */
        /* code the run length via per-position unary on run_pos */
        wubu_cabac_encode_bin(e,&st->lesc,0);  /* not-escape: run < 15 */
        for(int i=0;i<run && i<14;i++)wubu_cabac_encode_bin(e,&st->run_pos[i],1);
        if(run<15)wubu_cabac_encode_bin(e,&st->run_pos[run],0);
        /* code the level at pos */
        enc_sf(e,st,scanned[pos]);
        pos++;
    }
    return 0;
}

int wubu_cc_decode_tokctx(WuBuCabacDec* d,WuBuCCState* st,int* scanned,int n){
    memset(scanned,0,sizeof(int)*(size_t)n);
    int pos=0;
    while(pos<n){
        /* decode run length */
        unsigned run;
        if(wubu_cabac_decode_bin(d,&st->lesc)){
            unsigned r=0;
            for(int b=0;b<6;b++)r|=((unsigned)dec_bypass_bit(d))<<b;
            run=r+15;
        }else{
            run=0;
            for(run=0;run<14;run++){
                if(!wubu_cabac_decode_bin(d,&st->run_pos[run]))break;
            }
        }
        pos+=run;
        if(pos>=n)break;  /* EOB: rest is zeros (already memset) */
        /* decode level at pos */
        scanned[pos]=dec_sf(d,st);
        pos++;
    }
    return 0;
}

/* ===== Scheme C: lastpos ===== */
long wubu_cc_encode_lastpos(WuBuCabacEnc* e,WuBuCCState* st,const int* scanned,int n){
    int last=n-1;
    while(last>=0&&scanned[last]==0)last--;
    enc_u_count(e,st,(unsigned)(last+1));
    for(int i=0;i<=last;i++)enc_sf(e,st,scanned[i]);
    return 0;
}

int wubu_cc_decode_lastpos(WuBuCabacDec* d,WuBuCCState* st,int* scanned,int n){
    memset(scanned,0,sizeof(int)*(size_t)n);
    unsigned cnt=dec_u_count(d,st);
    if(cnt>(unsigned)n)return -1;
    for(unsigned i=0;i<cnt;i++)
        scanned[i]=dec_sf(d,st);
    return (int)cnt;
}

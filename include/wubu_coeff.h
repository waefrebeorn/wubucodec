/*
 * wubu_coeff.h -- C4: coefficient symbol alphabet schemes
 *
 * Persistent context state (WuBuCCState) must survive across blocks in a
 * frame — that's where CABAC beats fixed VLC: adaptive models warm up.
 */
#ifndef WUBU_COEFF_H
#define WUBU_COEFF_H
#include "wubu_cabac.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    CabacContext rctx[2];   /* run: early / late position (for tokctx run tokens) */
    CabacContext ectx;      /* more-data flag (lastpos count / tokctx run) */
    CabacContext xesc;     /* run escape (shared for runs and lastpos count) */
    CabacContext lesc;     /* level escape (for mag >= 15) */
    /* Per-position unary contexts — SEPARATE arrays for count vs level magnitude,
       because "is count > i?" and "is level mag > i?" have very different stats. */
    CabacContext count_pos[16];  /* count unary: "is significant-count > i?" */
    CabacContext level_pos[16];  /* level magnitude unary: "is |level| > i?" */
    CabacContext run_pos[16];    /* run-length unary (tokctx): "is runLen > i?" */
} WuBuCCState;

void wubu_cc_init_state(WuBuCCState* st);

/* Scheme B: interleaved run/level tokens with position-based contexts. */
long wubu_cc_encode_tokctx(WuBuCabacEnc* e,WuBuCCState* st,const int* scanned,int n);
int  wubu_cc_decode_tokctx(WuBuCabacDec* d,WuBuCCState* st,int* scanned,int n);

/* Scheme C: last-significant-count first, then all levels. */
long wubu_cc_encode_lastpos(WuBuCabacEnc* e,WuBuCCState* st,const int* scanned,int n);
int  wubu_cc_decode_lastpos(WuBuCabacDec* d,WuBuCCState* st,int* scanned,int n);

#ifdef __cplusplus
}
#endif
#endif

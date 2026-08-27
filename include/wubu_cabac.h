/* GROUP 3-6: CABAC binary arithmetic coder with context modeling */
#ifndef WUBU_CABAC_H
#define WUBU_CABAC_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t state;  /* 0..63 probability state index */
    uint8_t mps;    /* most probable symbol */
} CabacContext;

typedef struct {
    uint8_t* buf;
    size_t cap;
    size_t pos;
} WuBuCabacEnc;

void wubu_cabac_init_encoder(WuBuCabacEnc* e,uint8_t* buf,size_t cap);
void wubu_cabac_start(WuBuCabacEnc* e,uint8_t* buf,size_t cap);
void wubu_cabac_encode_bin(WuBuCabacEnc* e,CabacContext* ctx,int bin_val);
void wubu_cabac_encode_bypass(WuBuCabacEnc* e,int bin_val);
long wubu_cabac_finish(WuBuCabacEnc* e);
void wubu_cabac_init_contexts(CabacContext* contexts,int n_contexts,
                               int init_value,int qp);

/* ===== Decoder (C1) ===== */
typedef struct {
    uint8_t* buf;
    size_t cap;
    size_t pos;
} WuBuCabacDec;

void wubu_cabac_decoder_start(WuBuCabacDec* d,const uint8_t* buf,size_t bytes);
int  wubu_cabac_decode_bin(WuBuCabacDec* d,CabacContext* ctx);
int  wubu_cabac_decode_bypass(WuBuCabacDec* d);
long wubu_cabac_decoder_finish(WuBuCabacDec* d);

#ifdef __cplusplus
}
#endif
#endif

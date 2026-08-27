/* GAP-D005: hashed-bag text encoder front end for manifold CLIP */
#ifndef WUBU_TEXT_ENCODER_H
#define WUBU_TEXT_ENCODER_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    int out_dim;
    float* table;   /* [vocab_size, out_dim], deterministic init */
} WubuTextEncoder;

int  wubu_text_init(WubuTextEncoder* te,int vocab_size,int out_dim);
void wubu_text_free(WubuTextEncoder* te);
/* encode text -> out[out_dim] (deterministic; distinct texts differ) */
void wubu_text_encode(const WubuTextEncoder* te,const char* text,
                      float* out,int out_dim);
#ifdef __cplusplus
}
#endif
#endif

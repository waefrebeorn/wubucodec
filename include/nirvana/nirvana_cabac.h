/* nirvana_cabac.h — Context-Adaptive Binary Arithmetic Coder
 *
 * Designed for variable block sizes from the ground up.
 * Supports 4x4, 8x8, 16x16, 32x32, 64x64 blocks.
 *
 * Key features:
 * - Persistent state across blocks in a frame
 * - Adaptive probability estimation
 * - Context modeling based on position and neighbors
 * - Bypass coding for uniform distributions
 */
#ifndef NIRVANA_CABAC_H
#define NIRVANA_CABAC_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NIRVANA_CABAC_BITS 16
#define NIRVANA_CABAC_ONE (1 << NIRVANA_CABAC_BITS)
#define NIRVANA_CABAC_HALF (1 << (NIRVANA_CABAC_BITS - 1))
#define NIRVANA_CABAC_MAX_CONTEXTS 512

/* CABAC context state */
typedef struct {
    uint8_t pStateIdx;  /* Probability state index (0-63) */
    uint8_t valMPS;     /* Most probable symbol */
} NirvanaCABACContext;

/* CABAC encoder */
typedef struct {
    /* Context states */
    NirvanaCABACContext contexts[NIRVANA_CABAC_MAX_CONTEXTS];
    
    /* Arithmetic coding state */
    uint32_t low;
    uint32_t range;
    int pending_bits;
    
    /* Bit output */
    uint8_t byte_buffer;
    int byte_buffer_bits;
    uint8_t *output;
    int output_pos;
    int output_size;
    
    /* Position in frame */
    int block_x;
    int block_y;
    int block_size;
} NirvanaCABACEncoder;

/* CABAC decoder */
typedef struct {
    NirvanaCABACContext contexts[NIRVANA_CABAC_MAX_CONTEXTS];
    uint32_t low;
    uint32_t range;
    uint32_t value;
    const uint8_t *input;
    int input_pos;
    int input_size;
    int pending_bits;
    uint8_t byte_buffer;
    int byte_buffer_bits;
} NirvanaCABACDecoder;

/* Probability state transition tables */
static const uint8_t transIdxMPS[64] = {
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
    17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    49,50,51,52,53,54,55,56,57,58,59,60,61,62,62,63
};

static const uint8_t transIdxLPS[64] = {
    0,0,1,2,2,4,4,5,6,7,8,9,9,11,11,12,
    13,13,15,15,16,16,18,18,19,20,20,21,22,22,23,24,
    24,25,26,26,27,27,28,28,29,29,30,30,30,31,32,32,
    33,33,33,34,34,35,35,35,36,36,36,37,37,37,38,38
};

/* Initialize CABAC encoder */
static inline void nirvana_cabac_init(NirvanaCABACEncoder *enc, uint8_t *output, int size) {
    enc->low = 0;
    enc->range = NIRVANA_CABAC_ONE - 1;
    enc->pending_bits = 0;
    enc->byte_buffer = 0;
    enc->byte_buffer_bits = 0;
    enc->output = output;
    enc->output_pos = 0;
    enc->output_size = size;
    
    for (int i = 0; i < NIRVANA_CABAC_MAX_CONTEXTS; i++) {
        enc->contexts[i].pStateIdx = 32;
        enc->contexts[i].valMPS = 0;
    }
}

/* Write a bit to output */
static inline void nirvana_cabac_write_bit(NirvanaCABACEncoder *enc, int bit) {
    enc->byte_buffer = (enc->byte_buffer << 1) | bit;
    enc->byte_buffer_bits++;
    
    if (enc->byte_buffer_bits == 8) {
        if (enc->output_pos < enc->output_size) {
            enc->output[enc->output_pos++] = enc->byte_buffer;
        }
        enc->byte_buffer = 0;
        enc->byte_buffer_bits = 0;
    }
}

/* Flush remaining bits */
static inline void nirvana_cabac_flush_bits(NirvanaCABACEncoder *enc) {
    while (enc->byte_buffer_bits > 0) {
        enc->byte_buffer = (enc->byte_buffer << 1);
        enc->byte_buffer_bits++;
        if (enc->byte_buffer_bits == 8) {
            if (enc->output_pos < enc->output_size) {
                enc->output[enc->output_pos++] = enc->byte_buffer;
            }
            enc->byte_buffer = 0;
            enc->byte_buffer_bits = 0;
        }
    }
}

/* Encode a binary decision */
static inline void nirvana_cabac_encode_bin(NirvanaCABACEncoder *enc, int ctxIdx, int binVal) {
    NirvanaCABACContext *ctx = &enc->contexts[ctxIdx];
    uint8_t pState = ctx->pStateIdx;
    uint8_t valMPS = ctx->valMPS;
    
    /* Calculate LPS range */
    uint32_t rLPS = (pState < 32) ? (63 - pState) : pState;
    rLPS = (rLPS * enc->range) >> 8;
    if (rLPS < 2) rLPS = 2;
    
    uint32_t rMPS = enc->range - rLPS;
    
    /* Update range */
    if (binVal == valMPS) {
        enc->range = rMPS;
    } else {
        enc->low += rMPS;
        enc->range = rLPS;
    }
    
    /* Update context state */
    if (binVal == valMPS) {
        ctx->pStateIdx = transIdxMPS[pState];
    } else {
        ctx->pStateIdx = transIdxLPS[pState];
        if (pState == 0) ctx->valMPS = 1 - valMPS;
    }
    
    /* Renormalize */
    while (enc->range < NIRVANA_CABAC_HALF) {
        if (enc->low >= NIRVANA_CABAC_ONE) {
            enc->low -= NIRVANA_CABAC_ONE;
            nirvana_cabac_write_bit(enc, 1);
            while (enc->pending_bits > 0) {
                nirvana_cabac_write_bit(enc, 0);
                enc->pending_bits--;
            }
        } else {
            nirvana_cabac_write_bit(enc, 0);
            while (enc->pending_bits > 0) {
                nirvana_cabac_write_bit(enc, 1);
                enc->pending_bits--;
            }
        }
        
        enc->range <<= 1;
        enc->low <<= 1;
    }
}

/* Encode unsigned Exp-Golomb */
static inline void nirvana_cabac_encode_ue(NirvanaCABACEncoder *enc, int value, int ctxBase) {
    int prefix = 0;
    int temp = value;
    while (temp > 0) {
        nirvana_cabac_encode_bin(enc, ctxBase + prefix, 1);
        temp--;
        prefix++;
    }
    nirvana_cabac_encode_bin(enc, ctxBase + prefix, 0);
    
    if (prefix > 0) {
        for (int i = prefix - 1; i >= 0; i--) {
            nirvana_cabac_encode_bin(enc, ctxBase + 32 + i, (value >> i) & 1);
        }
    }
}

/* Encode signed Exp-Golomb */
static inline void nirvana_cabac_encode_se(NirvanaCABACEncoder *enc, int value, int ctxBase) {
    int absVal = abs(value);
    int sign = (value < 0) ? 1 : 0;
    nirvana_cabac_encode_ue(enc, absVal, ctxBase);
    if (absVal > 0) {
        nirvana_cabac_encode_bin(enc, ctxBase + 64, sign);
    }
}

/* Encode fixed-length binary */
static inline void nirvana_cabac_encode_bits(NirvanaCABACEncoder *enc, int value, int numBits, int ctxBase) {
    for (int i = numBits - 1; i >= 0; i--) {
        nirvana_cabac_encode_bin(enc, ctxBase + i, (value >> i) & 1);
    }
}

/* Flush encoder */
static inline int nirvana_cabac_finish(NirvanaCABACEncoder *enc) {
    enc->range = NIRVANA_CABAC_HALF / 2;
    enc->low += NIRVANA_CABAC_HALF / 2;
    
    while (enc->range < NIRVANA_CABAC_HALF) {
        if (enc->low >= NIRVANA_CABAC_ONE) {
            enc->low -= NIRVANA_CABAC_ONE;
            nirvana_cabac_write_bit(enc, 1);
        } else {
            nirvana_cabac_write_bit(enc, 0);
        }
        enc->range <<= 1;
        enc->low <<= 1;
    }
    
    nirvana_cabac_flush_bits(enc);
    return enc->output_pos;
}

/* Zigzag scan order generation */
static inline void nirvana_generate_zigzag(int *zigzag, int n) {
    int idx = 0;
    for (int s = 0; s < 2 * n - 1; s++) {
        if (s < n) {
            if (s % 2 == 0) {
                for (int r = s, c = 0; r >= 0 && c <= s; r--, c++) {
                    if (r < n && c < n) {
                        zigzag[idx++] = r * n + c;
                    }
                }
            } else {
                for (int r = 0, c = s; r <= s && c >= 0; r++, c--) {
                    if (r < n && c < n) {
                        zigzag[idx++] = r * n + c;
                    }
                }
            }
        } else {
            int d = s - n + 1;
            if (d % 2 == 0) {
                for (int r = d, c = n - 1; r < n && c >= d; r++, c--) {
                    if (r < n && c < n) {
                        zigzag[idx++] = r * n + c;
                    }
                }
            } else {
                for (int r = n - 1, c = d; r >= d && c < n; r--, c++) {
                    if (r < n && c < n) {
                        zigzag[idx++] = r * n + c;
                    }
                }
            }
        }
    }
}

#endif /* NIRVANA_CABAC_H */

/* wubu_cabac.c — Bit-packing CABAC encoder */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define NUM_CABAC_CONTEXTS 512
#define CABAC_BITS 16
#define CABAC_ONE (1 << CABAC_BITS)
#define CABAC_HALF (1 << (CABAC_BITS - 1))

typedef struct {
    uint8_t pStateIdx;
    uint8_t valMPS;
} CABACContext;

typedef struct {
    CABACContext contexts[NUM_CABAC_CONTEXTS];
    uint32_t low;
    uint32_t range;
    int pending_bits;
    uint8_t byte_buffer;
    int byte_buffer_bits;
    uint8_t* output;
    int output_pos;
    int output_size;
} CABACEncoder;

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

static void cabac_init(CABACEncoder* enc, uint8_t* output, int size) {
    enc->low = 0;
    enc->range = CABAC_ONE - 1;
    enc->pending_bits = 0;
    enc->byte_buffer = 0;
    enc->byte_buffer_bits = 0;
    enc->output = output;
    enc->output_pos = 0;
    enc->output_size = size;
    for (int i = 0; i < NUM_CABAC_CONTEXTS; i++) {
        enc->contexts[i].pStateIdx = 32;
        enc->contexts[i].valMPS = 0;
    }
}

static void cabac_write_bit(CABACEncoder* enc, int bit) {
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

static void cabac_flush_bits(CABACEncoder* enc) {
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

static void cabac_encode_decision(CABACEncoder* enc, int ctxIdx, int binVal) {
    CABACContext* ctx = &enc->contexts[ctxIdx];
    uint8_t pState = ctx->pStateIdx;
    uint8_t valMPS = ctx->valMPS;
    
    uint32_t rLPS = (pState < 32) ? (63 - pState) : pState;
    rLPS = (rLPS * enc->range) >> 8;
    if (rLPS < 2) rLPS = 2;
    
    uint32_t rMPS = enc->range - rLPS;
    
    if (binVal == valMPS) {
        enc->range = rMPS;
    } else {
        enc->low += rMPS;
        enc->range = rLPS;
    }
    
    if (binVal == valMPS) {
        ctx->pStateIdx = transIdxMPS[pState];
    } else {
        ctx->pStateIdx = transIdxLPS[pState];
        if (pState == 0) ctx->valMPS = 1 - valMPS;
    }
    
    while (enc->range < CABAC_HALF) {
        if (enc->low >= CABAC_ONE) {
            enc->low -= CABAC_ONE;
            cabac_write_bit(enc, 1);
            while (enc->pending_bits > 0) {
                cabac_write_bit(enc, 0);
                enc->pending_bits--;
            }
        } else {
            cabac_write_bit(enc, 0);
            while (enc->pending_bits > 0) {
                cabac_write_bit(enc, 1);
                enc->pending_bits--;
            }
        }
        
        enc->range <<= 1;
        enc->low <<= 1;
    }
}

static void cabac_encode_ue(CABACEncoder* enc, int value, int ctxBase) {
    int prefix = 0;
    int temp = value;
    while (temp > 0) {
        cabac_encode_decision(enc, ctxBase + prefix, 1);
        temp--;
        prefix++;
    }
    cabac_encode_decision(enc, ctxBase + prefix, 0);
    
    if (prefix > 0) {
        for (int i = prefix - 1; i >= 0; i--) {
            cabac_encode_decision(enc, ctxBase + 32 + i, (value >> i) & 1);
        }
    }
}

static void cabac_encode_se(CABACEncoder* enc, int value, int ctxBase) {
    int absVal = abs(value);
    int sign = (value < 0) ? 1 : 0;
    cabac_encode_ue(enc, absVal, ctxBase);
    if (absVal > 0) {
        cabac_encode_decision(enc, ctxBase + 64, sign);
    }
}

static int cabac_flush(CABACEncoder* enc) {
    enc->range = CABAC_HALF / 2;
    enc->low += CABAC_HALF / 2;
    
    while (enc->range < CABAC_HALF) {
        if (enc->low >= CABAC_ONE) {
            enc->low -= CABAC_ONE;
            cabac_write_bit(enc, 1);
        } else {
            cabac_write_bit(enc, 0);
        }
        enc->range <<= 1;
        enc->low <<= 1;
    }
    
    cabac_flush_bits(enc);
    return enc->output_pos;
}

/* Encode block with CABAC */
int cabac_encode_block(int16_t* levels, int num_coeffs, uint8_t* output, int output_size) {
    CABACEncoder enc;
    cabac_init(&enc, output, output_size);
    
    /* Generate zigzag table for this block size */
    int zigzag[4096];
    int n = (int)sqrt((float)num_coeffs);
    if (n * n != num_coeffs) n = 8; /* fallback */
    
    for (int i = 0; i < num_coeffs; i++) {
        int r = i / n;
        int c = i % n;
        int zz_r = r + c;
        if (zz_r < n) {
            if (zz_r % 2 == 0) {
                zigzag[i] = zz_r * (zz_r + 1) / 2 + r;
            } else {
                zigzag[i] = zz_r * (zz_r + 1) / 2 + c;
            }
        } else {
            int d = 2 * n - 1 - zz_r;
            if (d % 2 == 0) {
                zigzag[i] = num_coeffs - 1 - (d * (d + 1) / 2 + (n - 1 - r));
            } else {
                zigzag[i] = num_coeffs - 1 - (d * (d + 1) / 2 + (n - 1 - c));
            }
        }
    }
    
    /* Clamp to valid range */
    for (int i = 0; i < num_coeffs; i++) {
        if (zigzag[i] < 0) zigzag[i] = 0;
        if (zigzag[i] >= num_coeffs) zigzag[i] = num_coeffs - 1;
    }
    
    int num_nonzero = 0;
    for (int i = 0; i < num_coeffs; i++) {
        if (levels[zigzag[i]] != 0) num_nonzero++;
    }
    
    cabac_encode_ue(&enc, num_nonzero, 0);
    
    if (num_nonzero == 0) {
        return cabac_flush(&enc);
    }
    
    int run = 0;
    for (int i = 0; i < num_coeffs; i++) {
        int zz = zigzag[i];
        int level = levels[zz];
        
        if (level != 0) {
            cabac_encode_ue(&enc, run, 64);
            cabac_encode_se(&enc, level, 128);
            run = 0;
        } else {
            run++;
        }
    }
    
    return cabac_flush(&enc);
}

/* Estimate bits for CABAC encoding */
int cabac_bits_estimate(int16_t* levels, int num_coeffs) {
    /* Generate zigzag table for this block size */
    int zigzag[4096];
    int n = (int)sqrt((float)num_coeffs);
    if (n * n != num_coeffs) n = 8;
    
    for (int i = 0; i < num_coeffs; i++) {
        int r = i / n;
        int c = i % n;
        int zz_r = r + c;
        if (zz_r < n) {
            if (zz_r % 2 == 0) {
                zigzag[i] = zz_r * (zz_r + 1) / 2 + r;
            } else {
                zigzag[i] = zz_r * (zz_r + 1) / 2 + c;
            }
        } else {
            int d = 2 * n - 1 - zz_r;
            if (d % 2 == 0) {
                zigzag[i] = num_coeffs - 1 - (d * (d + 1) / 2 + (n - 1 - r));
            } else {
                zigzag[i] = num_coeffs - 1 - (d * (d + 1) / 2 + (n - 1 - c));
            }
        }
    }
    
    for (int i = 0; i < num_coeffs; i++) {
        if (zigzag[i] < 0) zigzag[i] = 0;
        if (zigzag[i] >= num_coeffs) zigzag[i] = num_coeffs - 1;
    }
    
    int bits = 0;
    int num_nonzero = 0;
    
    for (int i = 0; i < num_coeffs; i++) {
        if (levels[zigzag[i]] != 0) num_nonzero++;
    }
    
    /* Exp-golomb bits for num_nonzero */
    int k = 0;
    while ((1 << k) <= num_nonzero) k++;
    bits += 2 * k - 1;
    
    if (num_nonzero == 0) return bits;
    
    int run = 0;
    for (int i = 0; i < num_coeffs; i++) {
        int zz = zigzag[i];
        int level = levels[zz];
        
        if (level != 0) {
            /* Run bits */
            int rk = 0;
            while ((1 << rk) <= run) rk++;
            bits += 2 * rk - 1;
            
            /* Level bits */
            int mag = abs(level);
            int lk = 0;
            while ((1 << lk) <= mag) lk++;
            bits += 2 * lk - 1;
            bits += 1; /* Sign */
            
            run = 0;
        } else {
            run++;
        }
    }
    
    return bits;
}

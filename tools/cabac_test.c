/* cabac_test.c — Minimal CABAC test */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CABAC_BITS 16
#define CABAC_ONE (1 << CABAC_BITS)
#define CABAC_HALF (1 << (CABAC_BITS - 1))

typedef struct {
    uint32_t low, range;
    int pending_bits;
    uint8_t byte_buffer;
    int byte_buffer_bits;
    uint8_t *output;
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

static void cabac_init(CABACEncoder *enc, uint8_t *output, int size) {
    enc->low = 0;
    enc->range = CABAC_ONE - 1;
    enc->pending_bits = 0;
    enc->byte_buffer = 0;
    enc->byte_buffer_bits = 0;
    enc->output = output;
    enc->output_pos = 0;
    enc->output_size = size;
}

static void cabac_write_bit(CABACEncoder *enc, int bit) {
    enc->byte_buffer = (enc->byte_buffer << 1) | bit;
    enc->byte_buffer_bits++;
    if (enc->byte_buffer_bits == 8) {
        if (enc->output_pos < enc->output_size) enc->output[enc->output_pos++] = enc->byte_buffer;
        enc->byte_buffer = 0;
        enc->byte_buffer_bits = 0;
    }
}

static void cabac_flush_bits(CABACEncoder *enc) {
    while (enc->byte_buffer_bits > 0) {
        enc->byte_buffer = (enc->byte_buffer << 1);
        enc->byte_buffer_bits++;
        if (enc->byte_buffer_bits == 8) {
            if (enc->output_pos < enc->output_size) enc->output[enc->output_pos++] = enc->byte_buffer;
            enc->byte_buffer = 0;
            enc->byte_buffer_bits = 0;
        }
    }
}

static void cabac_encode_bin(CABACEncoder *enc, int ctxIdx, int binVal) {
    /* Simplified: use a single context model */
    static uint8_t ctx_mps = 0;
    static uint8_t ctx_state = 32;
    
    uint8_t valMPS = ctx_mps;
    uint8_t pState = ctx_state;
    
    uint32_t rLPS = (pState < 32) ? (63 - pState) : pState;
    rLPS = (rLPS * enc->range) >> 8;
    if (rLPS < 2) rLPS = 2;
    uint32_t rMPS = enc->range - rLPS;
    
    if (binVal == valMPS) { enc->range = rMPS; }
    else { enc->low += rMPS; enc->range = rLPS; }
    
    if (binVal == valMPS) { ctx_state = transIdxMPS[pState]; }
    else { ctx_state = transIdxLPS[pState]; if (pState == 0) ctx_mps = 1 - valMPS; }
    
    while (enc->range < CABAC_HALF) {
        if (enc->low >= CABAC_ONE) {
            enc->low -= CABAC_ONE;
            cabac_write_bit(enc, 1);
            while (enc->pending_bits > 0) { cabac_write_bit(enc, 0); enc->pending_bits--; }
        } else {
            cabac_write_bit(enc, 0);
            while (enc->pending_bits > 0) { cabac_write_bit(enc, 1); enc->pending_bits--; }
        }
        enc->range <<= 1;
        enc->low <<= 1;
    }
}

static void cabac_encode_ue(CABACEncoder *enc, int value) {
    int prefix = 0, temp = value;
    while (temp > 0) { cabac_encode_bin(enc, 0, 1); temp--; prefix++; }
    cabac_encode_bin(enc, 0, 0);
    if (prefix > 0) for (int i = prefix - 1; i >= 0; i--)
        cabac_encode_bin(enc, 0, (value >> i) & 1);
}

static void cabac_encode_se(CABACEncoder *enc, int value) {
    int absVal = abs(value), sign = (value < 0) ? 1 : 0;
    cabac_encode_ue(enc, absVal);
    if (absVal > 0) cabac_encode_bin(enc, 0, sign);
}

static int cabac_finish(CABACEncoder *enc) {
    enc->range = CABAC_HALF / 2;
    enc->low += CABAC_HALF / 2;
    while (enc->range < CABAC_HALF) {
        if (enc->low >= CABAC_ONE) { enc->low -= CABAC_ONE; cabac_write_bit(enc, 1); }
        else cabac_write_bit(enc, 0);
        enc->range <<= 1; enc->low <<= 1;
    }
    cabac_flush_bits(enc);
    return enc->output_pos;
}

int main() {
    printf("CABAC test\n");
    
    uint8_t output[1024*1024];
    CABACEncoder cabac;
    cabac_init(&cabac, output, sizeof(output));
    
    /* Encode some test values */
    int test_values[] = {0, 1, 2, 3, 5, 10, 100, 1000, -1, -5, -100};
    int n = sizeof(test_values) / sizeof(test_values[0]);
    
    for (int i = 0; i < n; i++) {
        cabac_encode_se(&cabac, test_values[i]);
        printf("Encoded %d\n", test_values[i]);
    }
    
    int bytes = cabac_finish(&cabac);
    printf("Total bytes: %d\n", bytes);
    
    return 0;
}

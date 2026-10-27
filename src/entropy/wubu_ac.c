/* wubu_ac.c — Fixed arithmetic coder
 *
 * The previous version had a broken bit-packer (static variable).
 * This version actually packs bits correctly.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define AC_BITS 16
#define AC_ONE (1 << AC_BITS)
#define AC_HALF (1 << (AC_BITS - 1))
#define AC_QUARTER (1 << (AC_BITS - 2))

typedef struct {
    uint32_t low;
    uint32_t range;
    int pending;
    uint8_t* output;
    int output_pos;
    int output_bit_pos;
    int output_size;
    
    /* Context models */
    uint16_t freq[128][2];
    uint16_t total[128];
} ArithmeticEncoder;

static void ac_init(ArithmeticEncoder* ac, uint8_t* output, int size) {
    ac->low = 0;
    ac->range = AC_ONE - 1;
    ac->pending = 0;
    ac->output = output;
    ac->output_pos = 0;
    ac->output_bit_pos = 0;
    ac->output_size = size;
    
    for (int i = 0; i < 128; i++) {
        ac->freq[i][0] = 1;
        ac->freq[i][1] = 1;
        ac->total[i] = 2;
    }
}

static void ac_write_bit(ArithmeticEncoder* ac, int bit) {
    if (ac->output_pos >= ac->output_size) return;
    
    if (bit) {
        ac->output[ac->output_pos] |= (1 << (7 - ac->output_bit_pos));
    }
    
    ac->output_bit_pos++;
    if (ac->output_bit_pos == 8) {
        ac->output_bit_pos = 0;
        ac->output_pos++;
    }
}

static void ac_encode_symbol(ArithmeticEncoder* ac, int ctx, int symbol) {
    uint32_t total = ac->total[ctx];
    uint32_t freq0 = ac->freq[ctx][0];
    
    uint32_t split = ac->low + (ac->range * freq0 + total - 1) / total;
    
    if (symbol == 0) {
        ac->range = split - ac->low;
    } else {
        ac->low = split;
    }
    
    ac->freq[ctx][symbol]++;
    ac->total[ctx]++;
    
    if (ac->total[ctx] > 1024) {
        ac->freq[ctx][0] = (ac->freq[ctx][0] + 1) / 2;
        ac->freq[ctx][1] = (ac->freq[ctx][1] + 1) / 2;
        ac->total[ctx] = ac->freq[ctx][0] + ac->freq[ctx][1];
    }
    
    /* Renormalize */
    while (ac->range < AC_HALF) {
        if (ac->low >= AC_ONE) {
            ac->low -= AC_ONE;
            ac_write_bit(ac, 1);
            while (ac->pending > 0) {
                ac_write_bit(ac, 0);
                ac->pending--;
            }
        } else if (ac->low + ac->range <= AC_ONE) {
            ac_write_bit(ac, 0);
            while (ac->pending > 0) {
                ac_write_bit(ac, 1);
                ac->pending--;
            }
        } else {
            ac->low -= AC_QUARTER;
            ac->pending++;
        }
        
        ac->low <<= 1;
        ac->range <<= 1;
    }
}

static void ac_encode_ue(ArithmeticEncoder* ac, int value, int ctx) {
    int prefix = 0;
    int temp = value;
    while (temp > 0) {
        ac_encode_symbol(ac, ctx, 1);
        temp--;
        prefix++;
    }
    ac_encode_symbol(ac, ctx, 0);
    
    if (prefix > 0) {
        for (int i = prefix - 1; i >= 0; i--) {
            ac_encode_symbol(ac, ctx + 16 + i, (value >> i) & 1);
        }
    }
}

static void ac_encode_se(ArithmeticEncoder* ac, int value, int ctx) {
    int absVal = abs(value);
    int sign = (value < 0) ? 1 : 0;
    ac_encode_ue(ac, absVal, ctx);
    if (absVal > 0) {
        ac_encode_symbol(ac, ctx + 8, sign);
    }
}

static int ac_flush(ArithmeticEncoder* ac) {
    ac->low += AC_QUARTER;
    ac->range = AC_QUARTER;
    
    while (ac->range < AC_HALF) {
        if (ac->low >= AC_ONE) {
            ac->low -= AC_ONE;
            ac_write_bit(ac, 1);
        } else {
            ac_write_bit(ac, 0);
        }
        ac->low <<= 1;
        ac->range <<= 1;
    }
    
    /* Flush remaining bits */
    if (ac->output_bit_pos > 0) {
        ac->output_pos++;
    }
    
    return ac->output_pos;
}

/* Encode a block using arithmetic coding */
int ac_encode_block(int16_t* levels, int num_coeffs, uint8_t* output, int output_size) {
    ArithmeticEncoder ac;
    ac_init(&ac, output, output_size);
    
    static const int zigzag[64] = {
         0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    };
    
    /* Count non-zero */
    int num_nonzero = 0;
    for (int i = 0; i < num_coeffs; i++) {
        if (levels[zigzag[i]] != 0) num_nonzero++;
    }
    
    /* Encode num_nonzero */
    ac_encode_ue(&ac, num_nonzero, 0);
    
    if (num_nonzero == 0) return ac_flush(&ac);
    
    /* Encode coefficients */
    int ctx = 10;
    int run = 0;
    
    for (int i = 0; i < num_coeffs; i++) {
        int zz = zigzag[i];
        int level = levels[zz];
        
        if (level != 0) {
            ac_encode_ue(&ac, run, ctx);
            ac_encode_se(&ac, level, ctx + 4);
            run = 0;
            ctx++;
        } else {
            run++;
        }
    }
    
    return ac_flush(&ac);
}

/* Estimate bits for arithmetic coding */
int ac_bits_estimate(int16_t* levels, int num_coeffs) {
    static const int zigzag[64] = {
         0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    };
    
    int bits = 0;
    int num_nonzero = 0;
    
    for (int i = 0; i < num_coeffs; i++) {
        if (levels[zigzag[i]] != 0) num_nonzero++;
    }
    
    /* Exp-golomb bits for num_nonzero */
    int k = 0;
    while ((1 << k) <= num_nonzero) k++;
    bits += (num_nonzero > 0) ? (2*k - 1) : 1;
    
    if (num_nonzero == 0) return bits;
    
    int run = 0;
    for (int i = 0; i < num_coeffs; i++) {
        int zz = zigzag[i];
        int level = levels[zz];
        
        if (level != 0) {
            int rk = 0;
            while ((1 << rk) <= run) rk++;
            bits += (run > 0) ? (2*rk - 1) : 1;
            
            int mag = abs(level);
            int lk = 0;
            while ((1 << lk) <= mag) lk++;
            bits += 2*lk;
            
            run = 0;
        } else {
            run++;
        }
    }
    
    return bits;
}

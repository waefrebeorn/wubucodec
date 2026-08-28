/* wubu_trellis.c — C10: Trellis Rate-Distortion Optimized Quantization (RDOQ)

GROUP 8: Trellis quantization (RDOQ)
G8.06: Rate-distortion optimized quantization via per-coefficient optimization.

For each DCT coefficient independently, tries candidate levels {0, floor(q), ceil(q)}
and picks the one minimizing D + λ·R where:
  - D = (coeff - level*qstep)^2  (squared error)
  - R = bits needed to code this level

Since our bit model is context-independent (bits depend only on |level|), the
per-coefficient decision IS the globally optimal trellis result.

Exit: trellis-RDOQ must beat or match rounding-based quantization on SSE+λ·bits.
*/

#define M_PI 3.14159265358979f
#include "wubu_trellis.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* bits needed to code a nonzero coefficient value */
/* Uses clz for speed */
int tr_bits_for_level(int level){
    if(level == 0) return 0;
    int s = level;
    unsigned code = (s > 0) ? (unsigned)(2*s - 1) : (unsigned)(-2*s);
    int k = 31 - __builtin_clz(code + 1);
    return 2*k + 1;
}

/* Fast trellize: skip RDOQ for coefficients that will clearly round to zero */
double wubu_trellis_quantize_fast(const double* coeffs,int n_coeffs,
                              double qstep,double lambda,
                              int16_t* output_levels){
    if(n_coeffs<=0||!coeffs||!output_levels)return 0;
    
    double total_rd=0.0;
    
    /* Threshold: if |coeff| < qstep/2, it rounds to zero — skip RDOQ */
    double half_q = qstep * 0.5;
    double rdoq_threshold = qstep * 1.5; /* only do full RDOQ for larger coeffs */
    
    for(int i=0;i<n_coeffs;i++){
        double c=coeffs[i];
        double abs_c = fabs(c);
        
        if(abs_c < half_q){
            /* Clearly rounds to zero — no RDOQ needed */
            output_levels[i] = 0;
            total_rd += c*c; /* distortion = coefficient squared */
            continue;
        }
        
        if(abs_c < rdoq_threshold){
            /* Fast path: just round, no RDOQ candidates */
            int level = (int)floor(c / qstep + (c >= 0.0 ? 0.5 : -0.5));
            output_levels[i] = (int16_t)level;
            double recon = (double)level * qstep;
            double d = c - recon;
            int bits = (level != 0) ? tr_bits_for_level(level) : 0;
            total_rd += d*d + lambda * bits;
            continue;
        }
        
        /* Full RDOQ for significant coefficients */
        double q=c/qstep;
        int16_t candidates[3];
        int nc=0;
        candidates[nc++]=0;
        int fl=(int)floor(q);
        int ce=(int)ceil(q);
        if(fl!=0)candidates[nc++]=fl;
        if(ce!=fl&&ce!=0)candidates[nc++]=ce;
        
        double best_rd=1e18;
        int best_c=0;
        for(int j=0;j<nc;j++){
            int level=candidates[j];
            double recon=(double)level*qstep;
            double d=c-recon;
            int bits=(level!=0)?tr_bits_for_level(level):0;
            double rd=d*d+lambda*(double)bits;
            if(rd<best_rd){
                best_rd=rd;
                best_c=j;
            }
        }
        output_levels[i]=candidates[best_c];
        total_rd+=best_rd;
    }
    return total_rd;
}

/* Original full trellis for quality-critical paths */

/* distortion of coding original value x as level l */
static double tr_distortion(double coeff,double qstep,int level){
    double recon=(double)level*qstep;
    double d=coeff-recon;
    return d*d;
}

/*
 * Trellis-optimized quantization of one coefficient block.
 * For each coefficient independently, picks the level in {0, floor(q), ceil(q)}
 * that minimizes D + λ·R.
 *
 * Returns total RD cost. Writes optimal levels to output_levels.
 */
double wubu_trellis_quantize(const double* coeffs,int n_coeffs,
                              double qstep,double lambda,
                              int16_t* output_levels){
    if(n_coeffs<=0||!coeffs||!output_levels)return 0;

    double total_rd=0.0;

    for(int i=0;i<n_coeffs;i++){
        double c=coeffs[i];
        double q=c/(double)qstep;

        /* generate candidate levels: 0, floor(q), ceil(q) */
        int16_t candidates[3];
        int nc=0;

        /* always include zero */
        candidates[nc++]=0;

        int fl=(int)floor(q);
        int ce=(int)ceil(q);

        /* include floor if different from zero */
        if(fl!=0)candidates[nc++]=fl;

        /* include ceil if different from floor and zero */
        if(ce!=fl&&ce!=0)candidates[nc++]=ce;

        /* find best candidate minimizing D + λ·R */
        double best_rd=1e18;
        int best_c=0;

        for(int j=0;j<nc;j++){
            int level=candidates[j];
            double dist=tr_distortion(c,qstep,level);
            int bits=(level!=0)?tr_bits_for_level(level):0;
            double rd=dist+lambda*(double)bits;
            if(rd<best_rd){
                best_rd=rd;
                best_c=j;
            }
        }

        output_levels[i]=candidates[best_c];
        total_rd+=best_rd;
    }

    return total_rd;
}

/*
 * Compare trellis vs standard rounding on a block.
 * Returns total trellis RD cost.
 * Outputs: SSE and bits for both methods.
 */
long wubu_trellis_vs_rounding(const double* coeffs,int n_coeffs,
                              double qstep,double lambda,
                              long* out_sse_standard,long* out_bits_standard,
                              long* out_sse_trellis,long* out_bits_trellis){
    if(!coeffs||!out_sse_standard||!out_bits_standard||
       !out_sse_trellis||!out_bits_trellis)return -1;

    /* ---- standard rounding ---- */
    int16_t std_levels[256];
    *out_sse_standard=0;
    *out_bits_standard=0;
    for(int i=0;i<n_coeffs;i++){
        double q=coeffs[i]/(double)qstep;
        int level=(int)floor(q+(q>=0.0?0.5:-0.5));
        std_levels[i]=(int16_t)level;
        *out_sse_standard+=(long)tr_distortion(coeffs[i],level,qstep);
        if(level!=0)*out_bits_standard+=tr_bits_for_level(level);
    }

    /* ---- trellis ---- */
    int16_t tr_levels[256];
    double rd=wubu_trellis_quantize(coeffs,n_coeffs,qstep,lambda,tr_levels);

    *out_sse_trellis=0;
    *out_bits_trellis=0;
    for(int i=0;i<n_coeffs;i++){
        *out_sse_trellis+=(long)tr_distortion(coeffs[i],tr_levels[i],qstep);
        if(tr_levels[i]!=0)*out_bits_trellis+=tr_bits_for_level(tr_levels[i]);
    }

    return (long)rd;
}

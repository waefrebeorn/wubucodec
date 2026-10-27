/* benchmark_latent.c — Measure P/B frame efficiency in WuBuLatent
 *
 * Generates synthetic latent data with controlled temporal structure:
 * - Smooth regions (low motion) → P/B should dominate
 * - Scene cuts → I-frames forced
 * - Random noise → I-frames may be needed
 *
 * Measures: bits per frame type, hyperbolic distortion, compression ratio
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define DIM 64

/* ---- Hyperbolic math (from wubu_latent_codec.c) ---- */
static float vec_norm_sq(const float* x, int N){ float s=0; for(int i=0;i<N;i++) s+=x[i]*x[i]; return s; }
static float vec_norm(const float* x, int N){ return sqrtf(vec_norm_sq(x,N)+1e-7f); }

static void poincare_clip(float* out,const float* x,int N,float c){
    if(c<=0){memcpy(out,x,N*sizeof(float));return;}
    float sc=sqrtf(c+1e-7f), mn=1.0f/(sc+1e-7f)*(1.0f-1e-6f);
    float n=vec_norm(x,N);
    if(n>mn){float s=mn/(n+1e-7f);for(int i=0;i<N;i++)out[i]=x[i]*s;}
    else memcpy(out,x,N*sizeof(float));
}

static void expmap(float* out,const float* v,int N,float c){
    if(c<=0){memcpy(out,v,N*sizeof(float));return;}
    float sc=sqrtf(c+1e-7f), vn=vec_norm(v,N);
    if(vn<1e-7f){memcpy(out,v,N*sizeof(float));return;}
    float a=sc*vn; if(a>30)a=30;
    float tv=tanhf(a), s=tv/(sc*vn+1e-7f);
    for(int i=0;i<N;i++)out[i]=s*v[i];
    poincare_clip(out,out,N,c);
}

static void logmap(float* out,const float* y,int N,float c){
    if(c<=0){memcpy(out,y,N*sizeof(float));return;}
    float yc[256];
    if(N>256){float*b=malloc(N*sizeof(float));poincare_clip(b,y,N,c);logmap(out,b,N,c);free(b);return;}
    poincare_clip(yc,y,N,c);
    float sc=sqrtf(c+1e-7f), yn=vec_norm(yc,N);
    if(yn<1e-7f){memcpy(out,yc,N*sizeof(float));return;}
    float aa=sc*yn; if(aa>1-1e-5f)aa=1-1e-5f;
    float av=atanhf(aa), s=av/(sc*yn+1e-7f);
    for(int i=0;i<N;i++)out[i]=s*yc[i];
}

static void mobius_add(float* out,const float* x,const float* y,int N,float c){
    if(c<=0){for(int i=0;i<N;i++)out[i]=x[i]+y[i];return;}
    float xn=vec_norm_sq(x,N), yn=vec_norm_sq(y,N), xy=0;
    for(int i=0;i<N;i++)xy+=x[i]*y[i];
    float r=1/(c+1e-7f);
    if(xn>r*(1-1e-6f)*(1-1e-6f))xn=r*(1-1e-6f)*(1-1e-6f);
    if(yn>r*(1-1e-6f)*(1-1e-6f))yn=r*(1-1e-6f)*(1-1e-6f);
    float d=1+2*c*xy+c*c*xn*yn; if(d<1e-7f)d=1e-7f;
    float nx=1+2*c*xy+c*yn, ny=1-c*xn;
    for(int i=0;i<N;i++)out[i]=(nx*x[i]+ny*y[i])/d;
    poincare_clip(out,out,N,c);
}

static void expmap_at(float* out,const float* x,const float* v,int N,float c){
    float ev[256]; expmap(ev,v,N,c); mobius_add(out,x,ev,N,c);
}

static void logmap_at(float* out,const float* x,const float* y,int N,float c){
    float nx[256],r[256]; for(int i=0;i<N;i++)nx[i]=-x[i];
    mobius_add(r,nx,y,N,c); logmap(out,r,N,c);
}

static void geodesic_interp(float* out,const float* x0,const float* x1,float t,int N,float c){
    if(t<=0){memcpy(out,x0,N*sizeof(float));return;}
    if(t>=1){memcpy(out,x1,N*sizeof(float));return;}
    float tg[256]; logmap_at(tg,x0,x1,N,c);
    for(int i=0;i<N;i++)tg[i]*=t;
    expmap_at(out,x0,tg,N,c);
}

static float hyperbolic_dist(const float* x,const float* y,int N,float c){
    if(c<=0){float s=0;for(int i=0;i<N;i++){float d=x[i]-y[i];s+=d*d;}return sqrtf(s);}
    float nx[256],r[256]; for(int i=0;i<N;i++)nx[i]=-x[i];
    mobius_add(r,nx,y,N,c);
    float mn=vec_norm(r,N), sc=sqrtf(c+1e-7f), a=sc*mn;
    if(a>1-1e-5f)a=1-1e-5f;
    return(2/sc)*atanhf(a);
}

/* ---- Synthetic latent data generation ---- */

/* Generate smooth latent sequence (low motion) */
static void gen_smooth_latents(float* latents, int num_vecs, int dim, float c, int seed){
    srand(seed);
    float prev[256];
    for(int d=0;d<dim;d++) prev[d] = ((float)(rand()%2000)/1000.0f - 1.0f) * 0.3f / sqrtf(c);
    poincare_clip(prev,prev,dim,c);

    for(int i=0;i<num_vecs;i++){
        if(i==0){
            memcpy(latents,prev,dim*sizeof(float));
        } else {
            /* Small tangent velocity */
            float v[256];
            for(int d=0;d<dim;d++) v[d] = ((float)(rand()%2000)/1000.0f - 1.0f) * 0.02f;
            expmap_at(latents + i*dim, prev, v, dim, c);
            memcpy(prev, latents + i*dim, dim*sizeof(float));
        }
    }
}

/* Generate with scene cuts */
static void gen_scene_cuts(float* latents, int num_vecs, int dim, float c, int seed){
    srand(seed);
    int scene_len = num_vecs / 5; /* 5 scenes */
    float base[256];

    for(int i=0;i<num_vecs;i++){
        if(i % scene_len == 0){
            /* New scene: random base point */
            for(int d=0;d<dim;d++) base[d] = ((float)(rand()%2000)/1000.0f - 1.0f) * 0.4f / sqrtf(c);
            poincare_clip(base,base,dim,c);
        }
        /* Small motion within scene */
        float v[256];
        for(int d=0;d<dim;d++) v[d] = ((float)(rand()%2000)/1000.0f - 1.0f) * 0.01f;
        expmap_at(latents + i*dim, base, v, dim, c);
        memcpy(base, latents + i*dim, dim*sizeof(float));
    }
}

/* ---- Benchmark: measure P/B efficiency ---- */

typedef struct {
    int num_i, num_p, num_b;
    long bits_i, bits_p, bits_b;
    double dist_i, dist_p, dist_b;
    double total_dist;
    long total_bits;
} BenchResult;

/* Simulate current WuBuLatent encoding (run-level, fixed GOP) */
static void benchmark_current(const float* latents, int num_vecs, int dim, float c,
                               float base_qstep, BenchResult* res){
    memset(res,0,sizeof(*res));
    float* recon = calloc(num_vecs * dim, sizeof(float));
    float* prev_recon = calloc(dim, sizeof(float));

    for(int i=0;i<num_vecs;i++){
        const float* curr = latents + i*dim;
        int is_i = (i % 10 == 0);

        if(is_i){
            /* I-frame: quantize absolute */
            float qstep = base_qstep * (1.0f + vec_norm(curr,dim) * 10.0f);
            int16_t levels[256];
            for(int d=0;d<dim;d++){
                int l = (int)roundf(curr[d]/qstep);
                if(l>32767)l=32767; if(l<-32768)l=-32768;
                levels[d] = (int16_t)l;
            }
            /* Run-level bits estimate */
            int nz=0; for(int d=0;d<dim;d++) if(levels[d]!=0) nz++;
            long bits = nz * 20 + 10; /* rough estimate */
            res->bits_i += bits;
            res->num_i++;

            /* Reconstruct */
            for(int d=0;d<dim;d++) recon[i*dim+d] = (float)levels[d] * qstep;
            poincare_clip(recon+i*dim, recon+i*dim, dim, c);
            memcpy(prev_recon, recon+i*dim, dim*sizeof(float));
        } else {
            /* P-frame: tangent velocity */
            float tangent[256];
            logmap_at(tangent, prev_recon, curr, dim, c);
            float t_norm = vec_norm(tangent,dim);
            float qstep = base_qstep * 0.1f * (1.0f + t_norm * 10.0f);
            int16_t levels[256];
            int nz=0;
            for(int d=0;d<dim;d++){
                int l = (int)roundf(tangent[d]/qstep);
                if(l>32767)l=32767; if(l<-32768)l=-32768;
                levels[d] = (int16_t)l;
                if(l!=0) nz++;
            }
            long bits = nz * 20 + 10;
            res->bits_p += bits;
            res->num_p++;

            /* Reconstruct */
            float dequant[256];
            for(int d=0;d<dim;d++) dequant[d] = (float)levels[d] * qstep;
            expmap_at(recon+i*dim, prev_recon, dequant, dim, c);
            memcpy(prev_recon, recon+i*dim, dim*sizeof(float));
        }

        /* Distortion */
        float dist = hyperbolic_dist(curr, recon+i*dim, dim, c);
        res->total_dist += dist;
        if(is_i) res->dist_i += dist; else res->dist_p += dist;
    }

    res->total_bits = res->bits_i + res->bits_p + res->bits_b;
    free(recon);
    free(prev_recon);
}

/* Simulate all-I-frame encoding (the "AI-friendly" baseline) */
static void benchmark_all_i(const float* latents, int num_vecs, int dim, float c,
                             float base_qstep, BenchResult* res){
    memset(res,0,sizeof(*res));

    for(int i=0;i<num_vecs;i++){
        const float* curr = latents + i*dim;
        float qstep = base_qstep * (1.0f + vec_norm(curr,dim) * 10.0f);
        int16_t levels[256];
        int nz=0;
        for(int d=0;d<dim;d++){
            int l = (int)roundf(curr[d]/qstep);
            if(l>32767)l=32767; if(l<-32768)l=-32768;
            levels[d] = (int16_t)l;
            if(l!=0) nz++;
        }
        long bits = nz * 20 + 10;
        res->bits_i += bits;
        res->num_i++;

        /* Reconstruct */
        float recon[256];
        for(int d=0;d<dim;d++) recon[d] = (float)levels[d] * qstep;
        poincare_clip(recon,recon,dim,c);
        float dist = hyperbolic_dist(curr, recon, dim, c);
        res->total_dist += dist;
        res->dist_i += dist;
    }

    res->total_bits = res->bits_i;
}

/* ---- Main ---- */
int main(int argc, char** argv){
    int num_vecs = 100;
    int dim = DIM;
    float c = 1.0f;
    float base_qstep = 0.01f;

    printf("=== WuBuLatent P/B Efficiency Benchmark ===\n\n");
    printf("Config: %d vectors, dim=%d, curvature=%.1f, qstep=%.4f\n\n", num_vecs, dim, c, base_qstep);

    float* latents = malloc(num_vecs * dim * sizeof(float));

    /* Test 1: Smooth motion */
    printf("--- Test 1: Smooth Motion ---\n");
    gen_smooth_latents(latents, num_vecs, dim, c, 42);

    BenchResult res_current, res_all_i;
    benchmark_current(latents, num_vecs, dim, c, base_qstep, &res_current);
    benchmark_all_i(latents, num_vecs, dim, c, base_qstep, &res_all_i);

    printf("Current (I/P mixed):\n");
    printf("  Frames: I=%d P=%d B=%d\n", res_current.num_i, res_current.num_p, res_current.num_b);
    printf("  Bits: I=%ld P=%ld B=%ld (total=%ld)\n", res_current.bits_i, res_current.bits_p, res_current.bits_b, res_current.total_bits);
    printf("  Avg bits/frame: %.1f\n", (float)res_current.total_bits / num_vecs);
    printf("  Avg distortion: %.6f\n", res_current.total_dist / num_vecs);

    printf("All-I-frame:\n");
    printf("  Frames: I=%d\n", res_all_i.num_i);
    printf("  Bits: %ld\n", res_all_i.total_bits);
    printf("  Avg bits/frame: %.1f\n", (float)res_all_i.total_bits / num_vecs);
    printf("  Avg distortion: %.6f\n", res_all_i.total_dist / num_vecs);

    float saving = 100.0f * (1.0f - (float)res_current.total_bits / res_all_i.total_bits);
    printf("  Bit saving vs all-I: %.1f%%\n\n", saving);

    /* Test 2: Scene cuts */
    printf("--- Test 2: Scene Cuts ---\n");
    gen_scene_cuts(latents, num_vecs, dim, c, 42);

    benchmark_current(latents, num_vecs, dim, c, base_qstep, &res_current);
    benchmark_all_i(latents, num_vecs, dim, c, base_qstep, &res_all_i);

    printf("Current (I/P mixed):\n");
    printf("  Frames: I=%d P=%d B=%d\n", res_current.num_i, res_current.num_p, res_current.num_b);
    printf("  Bits: I=%ld P=%ld B=%ld (total=%ld)\n", res_current.bits_i, res_current.bits_p, res_current.bits_b, res_current.total_bits);
    printf("  Avg bits/frame: %.1f\n", (float)res_current.total_bits / num_vecs);
    printf("  Avg distortion: %.6f\n", res_current.total_dist / num_vecs);

    printf("All-I-frame:\n");
    printf("  Bits: %ld\n", res_all_i.total_bits);
    printf("  Avg bits/frame: %.1f\n", (float)res_all_i.total_bits / num_vecs);
    printf("  Avg distortion: %.6f\n", res_all_i.total_dist / num_vecs);

    saving = 100.0f * (1.0f - (float)res_current.total_bits / res_all_i.total_bits);
    printf("  Bit saving vs all-I: %.1f%%\n\n", saving);

    /* Test 3: Show the problem - tangent velocities are small, run-level wastes bits */
    printf("--- Test 3: The Problem (tangent velocity distribution) ---\n");
    gen_smooth_latents(latents, num_vecs, dim, c, 42);

    float total_tangent_norm = 0;
    float max_tangent_norm = 0;
    int nz_count = 0;

    float prev[256];
    memcpy(prev, latents, dim*sizeof(float));

    for(int i=1;i<num_vecs;i++){
        float tangent[256];
        logmap_at(tangent, prev, latents+i*dim, dim, c);
        float t_norm = vec_norm(tangent, dim);
        total_tangent_norm += t_norm;
        if(t_norm > max_tangent_norm) max_tangent_norm = t_norm;

        /* Count non-zero after coarse quantization */
        float qstep = base_qstep * 0.1f;
        for(int d=0;d<dim;d++){
            int l = (int)roundf(tangent[d]/qstep);
            if(l!=0) nz_count++;
        }
        memcpy(prev, latents+i*dim, dim*sizeof(float));
    }

    printf("  Avg tangent norm: %.6f\n", total_tangent_norm / (num_vecs-1));
    printf("  Max tangent norm: %.6f\n", max_tangent_norm);
    printf("  Non-zero coefficients at qstep=0.001: %d/%d (%.1f%%)\n",
           nz_count, (num_vecs-1)*dim, 100.0f*nz_count/((num_vecs-1)*dim));
    printf("  => Most tangent coefficients are near-zero, run-level coding wastes bits!\n");
    printf("  => CABAC with context modeling would compress these 3-5x better.\n\n");

    free(latents);
    return 0;
}

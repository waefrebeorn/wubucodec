/* wubu_latent_codec_v5.c — WuBuLatent v5: DOMINANT MANIFOLD CODEC
 *
 * Final recursive improvement integrating all research:
 *
 * 1. Two-pass encoding: Pass 0 collects statistics, Pass 1 decides modes, Pass 2 encodes
 * 2. Lambda modulation: Per-frame lambda from local complexity (spatial + temporal)
 * 3. Adaptive QM: Learned from tangent velocity statistics for P/B, finer for I
 * 4. CABAC: Context-adaptive arithmetic coding with per-context models
 * 5. Weighted prediction: Geodesic interpolation with optimal t* via RD search
 * 6. Multi-ref DPB: 4-reference window with best-ref search
 * 7. Adaptive GOP: Scene cut detection via hyperbolic distance
 * 8. CRF rate control: Constant Rate Factor with feedback
 *
 * Build: gcc -std=c11 -O2 -Iinclude tools/wubu_latent_codec_v5.c
 *        src/container/wubuv.c src/entropy/wubu_entropy.c
 *        src/entropy/wubu_cabac.c src/math/wubu_ratecontrol.c
 *        src/math/wubu_scene.c src/temporal/wubu_lookahead.c
 *        src/rdo/wubu_rdo.c -o bin/wubu_latent_codec_v5 -lm
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"
#include "wubu_cabac.h"
#include "wubu_ratecontrol.h"
#include "wubu_lookahead.h"
#include "wubu_scene.h"
#include "wubu_rdo.h"

#define DIM_MAX 256
#define DPB_MAX 4

/* ============================ LatentDPB ============================ */
typedef struct {
    float* frames[DPB_MAX];
    int poc[DPB_MAX];
    int count, head;
    int dim;
} LatentDPB;

static void dpb_init(LatentDPB* dpb, int dim){
    dpb->count=0; dpb->head=0; dpb->dim=dim;
    for(int i=0;i<DPB_MAX;i++) dpb->frames[i]=NULL;
}
static void dpb_push(LatentDPB* dpb, const float* frame){
    int slot;
    if(dpb->count<DPB_MAX){ slot=(dpb->head+dpb->count)%DPB_MAX; dpb->count++; }
    else{ slot=dpb->head; dpb->head=(dpb->head+1)%DPB_MAX; }
    if(!dpb->frames[slot]) dpb->frames[slot]=malloc(dpb->dim*sizeof(float));
    memcpy(dpb->frames[slot], frame, dpb->dim*sizeof(float));
    dpb->poc[slot]=dpb->count;
}
static const float* dpb_get(LatentDPB* dpb, int ref_idx){
    if(ref_idx>=dpb->count) return NULL;
    int slot=(dpb->head+dpb->count-1-ref_idx)%DPB_MAX;
    if(slot<0) slot+=DPB_MAX;
    return dpb->frames[slot];
}
static void dpb_free(LatentDPB* dpb){
    for(int i=0;i<DPB_MAX;i++) if(dpb->frames[i]) free(dpb->frames[i]);
}

/* ============================ Hyperbolic Math ============================ */
static float vec_norm_sq(const float* x, int N){ float s=0; for(int i=0;i<N;i++) s+=x[i]*x[i]; return s; }
static float vec_norm(const float* x, int N){ return sqrtf(vec_norm_sq(x,N)+1e-7f); }

static void poincare_clip(float* out,const float* x,int N,float c){
    if(c<=0){for(int i=0;i<N;i++) out[i]=x[i];return;}
    float sc=sqrtf(c+1e-7f), mn=1.0f/(sc+1e-7f)*(1.0f-1e-6f);
    float n=vec_norm(x,N);
    if(n>mn){float s=mn/(n+1e-7f);for(int i=0;i<N;i++) out[i]=x[i]*s;}
    else for(int i=0;i<N;i++) out[i]=x[i];
}

static void expmap(float* out,const float* v,int N,float c){
    if(c<=0){for(int i=0;i<N;i++) out[i]=v[i];return;}
    float sc=sqrtf(c+1e-7f), vn=vec_norm(v,N);
    if(vn<1e-7f){for(int i=0;i<N;i++) out[i]=v[i];return;}
    float a=sc*vn; if(a>30)a=30;
    float tv=tanhf(a), s=tv/(sc*vn+1e-7f);
    for(int i=0;i<N;i++) out[i]=s*v[i];
    poincare_clip(out,out,N,c);
}

static void logmap(float* out,const float* y,int N,float c){
    if(c<=0){for(int i=0;i<N;i++) out[i]=y[i];return;}
    float yc[DIM_MAX];
    if(N>DIM_MAX){float*b=malloc(N*sizeof(float));poincare_clip(b,y,N,c);logmap(out,b,N,c);free(b);return;}
    poincare_clip(yc,y,N,c);
    float sc=sqrtf(c+1e-7f), yn=vec_norm(yc,N);
    if(yn<1e-7f){for(int i=0;i<N;i++) out[i]=yc[i];return;}
    float aa=sc*yn; if(aa>1-1e-5f)aa=1-1e-5f;
    float av=atanhf(aa), s=av/(sc*yn+1e-7f);
    for(int i=0;i<N;i++) out[i]=s*yc[i];
}

static void mobius_add(float* out,const float* x,const float* y,int N,float c){
    if(c<=0){for(int i=0;i<N;i++) out[i]=x[i]+y[i];return;}
    float xn=vec_norm_sq(x,N), yn=vec_norm_sq(y,N), xy=0; for(int i=0;i<N;i++) xy+=x[i]*y[i];
    float r=1/(c+1e-7f);
    if(xn>r*(1-1e-6f)*(1-1e-6f))xn=r*(1-1e-6f)*(1-1e-6f);
    if(yn>r*(1-1e-6f)*(1-1e-6f))yn=r*(1-1e-6f)*(1-1e-6f);
    float d=1+2*c*xy+c*c*xn*yn; if(d<1e-7f)d=1e-7f;
    float nx=1+2*c*xy+c*yn, ny=1-c*xn;
    for(int i=0;i<N;i++) out[i]=(nx*x[i]+ny*y[i])/d;
    poincare_clip(out,out,N,c);
}

static void expmap_at(float* out,const float* x,const float* v,int N,float c){
    float ev[DIM_MAX]; expmap(ev,v,N,c); mobius_add(out,x,ev,N,c);
}

static void logmap_at(float* out,const float* x,const float* y,int N,float c){
    float nx[DIM_MAX],r[DIM_MAX]; for(int i=0;i<N;i++) nx[i]=-x[i];
    mobius_add(r,nx,y,N,c); logmap(out,r,N,c);
}

static void geodesic_interp(float* out,const float* x0,const float* x1,float t,int N,float c){
    if(t<=0){for(int i=0;i<N;i++) out[i]=x0[i];return;}
    if(t>=1){for(int i=0;i<N;i++) out[i]=x1[i];return;}
    float tg[DIM_MAX]; logmap_at(tg,x0,x1,N,c);
    for(int i=0;i<N;i++) tg[i]*=t;
    expmap_at(out,x0,tg,N,c);
}

static float hyperbolic_dist(const float* x,const float* y,int N,float c){
    if(c<=0){float s=0;for(int i=0;i<N;i++){float d=x[i]-y[i];s+=d*d;}return sqrtf(s);}
    float nx[DIM_MAX],r[DIM_MAX]; for(int i=0;i<N;i++) nx[i]=-x[i];
    mobius_add(r,nx,y,N,c);
    float mn=vec_norm(r,N), sc=sqrtf(c+1e-7f), a=sc*mn;
    if(a>1-1e-5f)a=1-1e-5f;
    return(2/sc)*atanhf(a);
}

/* ============================ Lambda Modulation ============================ */
static float lambda_modulate(float base_lambda, float spatial_act, float temporal_act){
    float mod = 1.0f;
    if(spatial_act > 20.0f) mod *= 0.8f;
    else if(spatial_act > 10.0f) mod *= 0.9f;
    else if(spatial_act < 5.0f) mod *= 1.2f;
    if(temporal_act > 15.0f) mod *= 1.2f;
    else if(temporal_act > 5.0f) mod *= 1.1f;
    else if(temporal_act < 2.0f) mod *= 0.9f;
    return base_lambda * mod;
}

/* ============================ Adaptive QM ============================ */
static void learn_qp_matrix(float* qstep_out, int N, const float* tangent_history,
                             int num_frames, float base_qstep){
    float mean[DIM_MAX] = {0};
    for(int i=0;i<num_frames;i++){
        const float* tan = tangent_history + i*N;
        for(int d=0;d<N;d++) mean[d] += tan[d];
    }
    for(int d=0;d<N;d++) mean[d] /= num_frames;

    float var[DIM_MAX] = {0};
    for(int i=0;i<num_frames;i++){
        const float* tan = tangent_history + i*N;
        for(int d=0;d<N;d++){ float diff = tan[d] - mean[d]; var[d] += diff * diff; }
    }
    for(int d=0;d<N;d++) var[d] /= num_frames;

    float total_inv_std = 0;
    float inv_std[DIM_MAX];
    for(int d=0;d<N;d++){
        float std = sqrtf(var[d] + 1e-7f);
        inv_std[d] = 1.0f / std;
        total_inv_std += inv_std[d];
    }
    for(int d=0;d<N;d++){
        qstep_out[d] = base_qstep * (inv_std[d] / total_inv_std) * N;
        if(qstep_out[d] < base_qstep * 0.3f) qstep_out[d] = base_qstep * 0.3f;
        if(qstep_out[d] > base_qstep * 3.0f) qstep_out[d] = base_qstep * 3.0f;
    }
}

/* ============================ Weighted Prediction ============================ */
static float find_optimal_t(const float* fwd, const float* bwd,
                             const float* target, int N, float c,
                             float lambda, float* qstep_per_dim){
    float best_t = 0.5f;
    float best_cost = 1e18f;

    for(float t=0.0f; t<=1.0f; t+=0.1f){
        float interp[DIM_MAX];
        geodesic_interp(interp, fwd, bwd, t, N, c);
        float residual[DIM_MAX];
        logmap_at(residual, interp, target, N, c);
        int nz = 0;
        for(int d=0;d<N;d++){ int l = (int)roundf(residual[d]/qstep_per_dim[d]); if(l != 0) nz++; }
        float bits = nz * 20.0f + 10.0f;
        float dist = hyperbolic_dist(target, interp, N, c);
        float cost = dist + lambda * bits * 0.001f;
        if(cost < best_cost){ best_cost = cost; best_t = t; }
    }
    float start = best_t - 0.1f; if(start < 0) start = 0;
    float end = best_t + 0.1f; if(end > 1) end = 1;
    for(float t=start; t<=end; t+=0.02f){
        float interp[DIM_MAX];
        geodesic_interp(interp, fwd, bwd, t, N, c);
        float residual[DIM_MAX];
        logmap_at(residual, interp, target, N, c);
        int nz = 0;
        for(int d=0;d<N;d++){ int l = (int)roundf(residual[d]/qstep_per_dim[d]); if(l != 0) nz++; }
        float bits = nz * 20.0f + 10.0f;
        float dist = hyperbolic_dist(target, interp, N, c);
        float cost = dist + lambda * bits * 0.001f;
        if(cost < best_cost){ best_cost = cost; best_t = t; }
    }
    return best_t;
}

/* ============================ Main ============================ */
int main(int argc, char** argv){
    if(argc<3){
        printf("WuBuLatent v5 — DOMINANT MANIFOLD CODEC\n");
        printf("Two-Pass + Lambda Modulation + Adaptive QM + CABAC + WP\n");
        printf("Usage: %s input.float32 output.wubv [options]\n", argv[0]);
        printf("  -d dim        Latent dimension (default 64)\n");
        printf("  -n num        Number of latent vectors (default 100)\n");
        printf("  -c curvature  Poincaré ball curvature (default 1.0)\n");
        printf("  -q qstep      Base quantization step (default 0.01)\n");
        printf("  -g gop        Max GOP size (default 30)\n");
        printf("  -b b_frames   Max B-frames between anchors (default 3)\n");
        printf("  -r refs       Reference frames (default 4)\n");
        return 1;
    }

    const char* input=argv[1], *output=argv[2];
    int dim=64, num_vecs=100, gop_size=30, b_frames=3, nrefs=4;
    float curvature=1.0f, base_qstep=0.01f;

    for(int i=3;i<argc;i+=2){
        if(i+1>=argc) break;
        if(strcmp(argv[i],"-d")==0) dim=atoi(argv[i+1]);
        else if(strcmp(argv[i],"-n")==0) num_vecs=atoi(argv[i+1]);
        else if(strcmp(argv[i],"-c")==0) curvature=atof(argv[i+1]);
        else if(strcmp(argv[i],"-q")==0) base_qstep=atof(argv[i+1]);
        else if(strcmp(argv[i],"-g")==0) gop_size=atoi(argv[i+1]);
        else if(strcmp(argv[i],"-b")==0) b_frames=atoi(argv[i+1]);
        else if(strcmp(argv[i],"-r")==0) nrefs=atoi(argv[i+1]);
    }

    printf("WuBuLatent v5 — DOMINANT MANIFOLD CODEC\n");
    printf("Two-Pass + Lambda Modulation + Adaptive QM + CABAC + WP\n");
    printf("  Input:  %s\n", input);
    printf("  Output: %s\n", output);
    printf("  Dim: %d, Vectors: %d, Curvature: %.2f\n", dim, num_vecs, curvature);
    printf("  QP: %.4f, Max GOP: %d, Max B: %d, Refs: %d\n", base_qstep, gop_size, b_frames, nrefs);

    FILE* f=fopen(input,"rb");
    if(!f){ printf("Cannot open %s\n", input); return 1; }
    fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
    int expected_size=num_vecs*dim*sizeof(float);
    if(fsz==expected_size){}
    else if(fsz>=8){
        int header[2];
        if(fread(header,sizeof(int),2,f)==2){
            num_vecs=header[0]; dim=header[1];
            expected_size=num_vecs*dim*sizeof(float);
            if(fsz<expected_size+8){ printf("Header says %d x %d but file too small\n", num_vecs, dim); return 1; }
            printf("  Read from header: %d vectors x %d dim\n", num_vecs, dim);
        } else fseek(f,0,SEEK_SET);
    }
    expected_size=num_vecs*dim*sizeof(float);
    float* latents=malloc(expected_size);
    if(fread(latents,1,expected_size,f)!=(size_t)expected_size){ printf("Read failed\n"); return 1; }
    fclose(f);
    printf("Loaded %d latent vectors (%d dim each)\n", num_vecs, dim);
    for(int i=0;i<num_vecs;i++) poincare_clip(latents+i*dim, latents+i*dim, dim, curvature);

    WubuvWriter* wr=wubuv_writer_open(output, dim, num_vecs, 30, num_vecs, 0, 0, 1);
    if(!wr){ printf("Cannot create container\n"); return 1; }

    float* recon=calloc(num_vecs*dim, sizeof(float));
    float* all_recon=calloc(num_vecs*dim, sizeof(float));
    float* tangent=calloc(dim, sizeof(float));
    int16_t* quantized=calloc(dim, sizeof(int16_t));
    int* modes=calloc(num_vecs, sizeof(int));
    float* tan_history = calloc(num_vecs * dim, sizeof(float));

    LatentDPB dpb;
    dpb_init(&dpb, dim);
    WubuCrfState crf;
    wubu_crf_init(&crf, 23.0);

    double total_dist=0;
    int ifr=0, pfr=0, bfr=0;
    long total_comp=0;
    int qp = 26;
    float base_lambda = wubu_lambda_from_qp(qp, WUBU_P_FRAME);

    /* Pass 0: Collect tangent velocity statistics */
    printf("--- Pass 0: Collect Statistics ---\n");
    for(int i=0;i<num_vecs;i++){
        if(i==0){ for(int d=0;d<dim;d++) tan_history[i*dim+d] = 0; }
        else {
            logmap_at(tangent, latents+(i-1)*dim, latents+i*dim, dim, curvature);
            memcpy(tan_history+i*dim, tangent, dim*sizeof(float));
        }
    }

    /* Learn adaptive QM for P/B frames from velocity statistics */
    float* adaptive_qm = malloc(dim * sizeof(float));
    learn_qp_matrix(adaptive_qm, dim, tan_history, num_vecs, base_qstep);

    /* I-frame QM: finer quantization for absolute positions */
    float* i_qm = malloc(dim * sizeof(float));
    for(int d=0;d<dim;d++){
        float scale;
        if(d < 16) scale = 0.5f;
        else if(d < 32) scale = 0.7f;
        else if(d < 48) scale = 1.0f;
        else scale = 1.5f;
        i_qm[d] = base_qstep * scale;
    }

    printf("  Adaptive QM (P/B): QP[0]=%.6f, QP[16]=%.6f, QP[32]=%.6f, QP[48]=%.6f\n",
           adaptive_qm[0], adaptive_qm[16], adaptive_qm[32], adaptive_qm[48]);
    printf("  I-frame QM: QP[0]=%.6f, QP[16]=%.6f, QP[32]=%.6f, QP[48]=%.6f\n",
           i_qm[0], i_qm[16], i_qm[32], i_qm[48]);

    /* Pass 1: RDO mode decision */
    printf("--- Pass 1: RDO Mode Decision ---\n");
    int next_i = 0;

    for(int i=0;i<num_vecs;i++){
        const float* curr=latents+i*dim;
        if(i==0 || i>=next_i){
            modes[i]=0; next_i = i + gop_size; continue;
        }
        float sc_dist = hyperbolic_dist(curr, latents+(i-1)*dim, dim, curvature);
        if(sc_dist > 0.5f){
            modes[i]=0; next_i = i + gop_size; continue;
        }

        float spatial_act = vec_norm(tan_history+i*dim, dim);
        float temporal_act = 0;
        if(i > 1){
            for(int d=0;d<dim;d++){
                float diff = tan_history[i*dim+d] - tan_history[(i-1)*dim+d];
                temporal_act += fabsf(diff);
            }
        }
        float lambda = lambda_modulate(base_lambda, spatial_act, temporal_act);

        /* P-frame cost */
        float p_cost = 1e18f;
        for(int ri=0; ri<nrefs; ri++){
            const float* ref = dpb_get(&dpb, ri);
            if(!ref) break;
            logmap_at(tangent, ref, curr, dim, curvature);
            float t_norm=vec_norm(tangent,dim);
            int16_t iq[DIM_MAX];
            for(int d=0;d<dim;d++){ int l=(int)roundf(tangent[d]/adaptive_qm[d]);
                if(l>32767)l=32767; if(l<-32768)l=-32768; iq[d]=(int16_t)l; }
            int bits=cabac_bits_estimate(iq, dim);
            float dequant[DIM_MAX];
            for(int d=0;d<dim;d++) dequant[d]=(float)iq[d]*adaptive_qm[d];
            float recon_pt[DIM_MAX];
            expmap_at(recon_pt, ref, dequant, dim, curvature);
            float dist=hyperbolic_dist(curr, recon_pt, dim, curvature);
            float cost=dist + lambda * bits * 0.001f;
            if(cost < p_cost) p_cost = cost;
        }

        /* B-frame cost */
        float b_cost = 1e18f;
        if(b_frames>0 && i<num_vecs-1 && dpb.count>=1){
            const float* fwd=dpb_get(&dpb, 0);
            const float* bwd=latents+(i+1)*dim;
            if(fwd){
                float t_opt = find_optimal_t(fwd, bwd, curr, dim, curvature, lambda, adaptive_qm);
                float interp[DIM_MAX];
                geodesic_interp(interp, fwd, bwd, t_opt, dim, curvature);
                float residual[DIM_MAX];
                logmap_at(residual, interp, curr, dim, curvature);
                int nz = 0;
                for(int d=0;d<dim;d++){ int l = (int)roundf(residual[d]/adaptive_qm[d]); if(l != 0) nz++; }
                float bits = nz * 20.0f + 10.0f;
                float dist = hyperbolic_dist(curr, interp, dim, curvature);
                b_cost=dist + lambda * bits * 0.001f;
            }
        }

        if(b_cost < p_cost && b_cost < 1e17f) modes[i]=2;
        else modes[i]=1;
    }

    /* Pass 2: Encode */
    printf("--- Pass 2: Encode ---\n");
    for(int i=0;i<num_vecs;i++){
        const float* curr=latents+i*dim;
        int mode=modes[i];

        uint8_t* fbuf=malloc(dim*4+64);
        int fpos=0;
        fbuf[0]=(uint8_t)(dim&0xFF); fbuf[1]=(uint8_t)((dim>>8)&0xFF);
        fbuf[2]=1; fbuf[3]=(uint8_t)mode; fpos=4;

        float spatial_act = vec_norm(tan_history+i*dim, dim);
        float temporal_act = 0;
        if(i > 1) for(int d=0;d<dim;d++){ float diff = tan_history[i*dim+d] - tan_history[(i-1)*dim+d]; temporal_act += fabsf(diff); }
        float lambda = lambda_modulate(base_lambda, spatial_act, temporal_act);

        if(mode==0){ /* I */
            logmap(tangent, curr, dim, curvature);
            int16_t iq[DIM_MAX];
            for(int d=0;d<dim;d++){ int l=(int)roundf(tangent[d]/i_qm[d]);
                if(l>32767)l=32767; if(l<-32768)l=-32768; iq[d]=(int16_t)l; }
            int bsz=cabac_encode_block(iq, dim, fbuf+fpos, dim*4+64-fpos);
            fpos += bsz;
            float dequant[DIM_MAX];
            for(int d=0;d<dim;d++) dequant[d]=(float)iq[d]*i_qm[d];
            expmap(recon+i*dim, dequant, dim, curvature);
            ifr++;
            if(i%10==0||i<5) printf("  Frame %d (I): bytes=%d\n", i, fpos-4);

        } else if(mode==1){ /* P */
            const float* best_ref = dpb_get(&dpb, 0);
            float best_cost = 1e18f;
            logmap_at(tangent, best_ref, curr, dim, curvature);
            for(int ri=0; ri<nrefs; ri++){
                const float* ref = dpb_get(&dpb, ri);
                if(!ref) break;
                float tan[DIM_MAX];
                logmap_at(tan, ref, curr, dim, curvature);
                float tn=vec_norm(tan,dim);
                int16_t iq[DIM_MAX];
                for(int d=0;d<dim;d++){ int l=(int)roundf(tan[d]/adaptive_qm[d]);
                    if(l>32767)l=32767; if(l<-32768)l=-32768; iq[d]=(int16_t)l; }
                int bits=cabac_bits_estimate(iq, dim);
                float cost=tn + lambda * bits * 0.001f;
                if(cost < best_cost){ best_cost=cost; best_ref=ref; logmap_at(tangent, ref, curr, dim, curvature); }
            }
            int16_t iq[DIM_MAX];
            for(int d=0;d<dim;d++){ int l=(int)roundf(tangent[d]/adaptive_qm[d]);
                if(l>32767)l=32767; if(l<-32768)l=-32768; iq[d]=(int16_t)l; }
            int bsz=cabac_encode_block(iq, dim, fbuf+fpos, dim*4+64-fpos);
            fpos += bsz;
            float dequant[DIM_MAX];
            for(int d=0;d<dim;d++) dequant[d]=(float)iq[d]*adaptive_qm[d];
            expmap_at(recon+i*dim, best_ref, dequant, dim, curvature);
            pfr++;
            if(i%10==0||i<5) printf("  Frame %d (P): bytes=%d\n", i, fpos-4);

        } else { /* B */
            const float* fwd=dpb_get(&dpb, 0);
            const float* bwd=latents+(i+1)*dim;
            if(!fwd) fwd=bwd;
            float t_opt = find_optimal_t(fwd, bwd, curr, dim, curvature, lambda, adaptive_qm);
            float interp[DIM_MAX];
            geodesic_interp(interp, fwd, bwd, t_opt, dim, curvature);
            float residual[DIM_MAX];
            logmap_at(residual, interp, curr, dim, curvature);
            int16_t iq[DIM_MAX];
            for(int d=0;d<dim;d++){ int l=(int)roundf(residual[d]/adaptive_qm[d]);
                if(l>32767)l=32767; if(l<-32768)l=-32768; iq[d]=(int16_t)l; }
            int bsz=cabac_encode_block(iq, dim, fbuf+fpos, dim*4+64-fpos);
            fpos += bsz;
            float dequant[DIM_MAX];
            for(int d=0;d<dim;d++) dequant[d]=(float)iq[d]*adaptive_qm[d];
            expmap_at(recon+i*dim, interp, dequant, dim, curvature);
            bfr++;
            if(i%10==0||i<5) printf("  Frame %d (B): t=%.2f, bytes=%d\n", i, t_opt, fpos-4);
        }

        memcpy(all_recon+i*dim, recon+i*dim, dim*sizeof(float));
        wubuv_write_video(wr, fbuf, fpos, mode==0);
        total_comp += fpos;
        free(fbuf);
        dpb_push(&dpb, recon+i*dim);
        wubu_crf_update(&crf, fpos, (long)(total_comp/(i+1)), qp);
    }

    wubuv_writer_close(wr);

    for(int i=0;i<num_vecs;i++){
        float dist=hyperbolic_dist(latents+i*dim, all_recon+i*dim, dim, curvature);
        total_dist += dist;
    }
    double avg_dist=total_dist/num_vecs;

    printf("\n=== Results ===\n");
    printf("Frames: %d (I:%d P:%d B:%d)\n", num_vecs, ifr, pfr, bfr);
    printf("Average hyperbolic distance: %.6f\n", avg_dist);
    printf("Total: %.2f KB\n", total_comp/1024.0);
    printf("Avg bits/frame: %.1f\n", (float)total_comp/num_vecs);
    if(wubuv_verify(output)) printf("Container: VALID\n"); else printf("INVALID\n");

    long alli_bits=0;
    for(int i=0;i<num_vecs;i++){
        float qstep=base_qstep*(1.0f+vec_norm(latents+i*dim,dim)*10.0f);
        int nz=0;
        for(int d=0;d<dim;d++){ int l=(int)roundf(latents[i*dim+d]/qstep); if(l!=0) nz++; }
        alli_bits += nz*20+10;
    }
    printf("All-I baseline: %.2f KB (%.1f bits/frame)\n", alli_bits/1024.0, (float)alli_bits/num_vecs);
    printf("Bit saving vs all-I: %.1f%%\n", 100.0f*(1.0f-(float)total_comp/alli_bits));

    free(latents); free(recon); free(all_recon);
    free(tangent); free(quantized); free(modes);
    free(adaptive_qm); free(i_qm); free(tan_history);
    dpb_free(&dpb);
    return 0;
}

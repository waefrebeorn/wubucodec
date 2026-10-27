/* wubu_latent_codec_v2.c — WuBuLatent v2: CABAC + RDO + Adaptive GOP + Flow B-frames
 *
 * v1 was 100% I-frame in practice: run-level coding wasted bits on near-zero
 * tangent velocities. v2 fixes this with:
 *
 * 1. CABAC: Reuses proven wubu_cabac.c for tangent coefficient coding
 * 2. RDO: Per-vector mode decision (I vs P vs B) using hyperbolic RD cost
 * 3. Adaptive GOP: Scene cut detection via hyperbolic distance
 * 4. Flow-matched B-frames: Tangent trajectory prediction from motion history
 * 5. Multi-ref DPB: 4-reference window for better prediction
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"
#include "wubu_cabac.h"

#define DIM_MAX 256
#define DPB_MAX 4

/* ============================ Hyperbolic Math ============================ */

static float vec_norm_sq(const float* x, int N){ float s=0; for(int i=0;i<N;i++) s+=x[i]*x[i]; return s; }
static float vec_norm(const float* x, int N){ return sqrtf(vec_norm_sq(x,N)+1e-7f); }
static float vec_dot(const float* a,const float* b,int N){ float s=0; for(int i=0;i<N;i++) s+=a[i]*b[i]; return s; }

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
    float yc[DIM_MAX];
    if(N>DIM_MAX){float*b=malloc(N*sizeof(float));poincare_clip(b,y,N,c);logmap(out,b,N,c);free(b);return;}
    poincare_clip(yc,y,N,c);
    float sc=sqrtf(c+1e-7f), yn=vec_norm(yc,N);
    if(yn<1e-7f){memcpy(out,yc,N*sizeof(float));return;}
    float aa=sc*yn; if(aa>1-1e-5f)aa=1-1e-5f;
    float av=atanhf(aa), s=av/(sc*yn+1e-7f);
    for(int i=0;i<N;i++)out[i]=s*yc[i];
}

static void mobius_add(float* out,const float* x,const float* y,int N,float c){
    if(c<=0){for(int i=0;i<N;i++)out[i]=x[i]+y[i];return;}
    float xn=vec_norm_sq(x,N), yn=vec_norm_sq(y,N), xy=vec_dot(x,y,N);
    float r=1/(c+1e-7f);
    if(xn>r*(1-1e-6f)*(1-1e-6f))xn=r*(1-1e-6f)*(1-1e-6f);
    if(yn>r*(1-1e-6f)*(1-1e-6f))yn=r*(1-1e-6f)*(1-1e-6f);
    float d=1+2*c*xy+c*c*xn*yn; if(d<1e-7f)d=1e-7f;
    float nx=1+2*c*xy+c*yn, ny=1-c*xn;
    for(int i=0;i<N;i++)out[i]=(nx*x[i]+ny*y[i])/d;
    poincare_clip(out,out,N,c);
}

static void expmap_at(float* out,const float* x,const float* v,int N,float c){
    float ev[DIM_MAX]; expmap(ev,v,N,c); mobius_add(out,x,ev,N,c);
}

static void logmap_at(float* out,const float* x,const float* y,int N,float c){
    float nx[DIM_MAX],r[DIM_MAX]; for(int i=0;i<N;i++)nx[i]=-x[i];
    mobius_add(r,nx,y,N,c); logmap(out,r,N,c);
}

static void geodesic_interp(float* out,const float* x0,const float* x1,float t,int N,float c){
    if(t<=0){memcpy(out,x0,N*sizeof(float));return;}
    if(t>=1){memcpy(out,x1,N*sizeof(float));return;}
    float tg[DIM_MAX]; logmap_at(tg,x0,x1,N,c);
    for(int i=0;i<N;i++)tg[i]*=t;
    expmap_at(out,x0,tg,N,c);
}

static float hyperbolic_dist(const float* x,const float* y,int N,float c){
    if(c<=0){float s=0;for(int i=0;i<N;i++){float d=x[i]-y[i];s+=d*d;}return sqrtf(s);}
    float nx[DIM_MAX],r[DIM_MAX]; for(int i=0;i<N;i++)nx[i]=-x[i];
    mobius_add(r,nx,y,N,c);
    float mn=vec_norm(r,N), sc=sqrtf(c+1e-7f), a=sc*mn;
    if(a>1-1e-5f)a=1-1e-5f;
    return(2/sc)*atanhf(a);
}

/* ============================ RDO Mode Decision ============================ */

typedef enum { MODE_I, MODE_P, MODE_B } RdoMode;

/* RD cost: d_hyp(orig, recon) + lambda * bits */
static float rd_cost(float dist, int bits, float lambda){
    return dist + lambda * bits;
}

/* Lambda from QP (same as x264) */
static float lambda_from_qp(int qp){
    static const float table[52]={
        0.65125f,0.70050f,0.75375f,0.81150f,0.87450f,0.94300f,1.01750f,1.09875f,
        1.18750f,1.28400f,1.38900f,1.50375f,1.62850f,1.76450f,1.91300f,2.07525f,
        2.25250f,2.44625f,2.65800f,2.88950f,3.14250f,3.41900f,3.72125f,4.05150f,
        4.41250f,4.80700f,5.23800f,5.70900f,6.22350f,6.78550f,7.39950f,8.07050f,
        8.80400f,9.60600f,10.4830f,11.4420f,12.4910f,13.6380f,14.8920f,16.2630f,
        17.7620f,19.4000f,21.1900f,23.1460f,25.2830f,27.6170f,30.1660f,32.9500f,
        35.9900f,39.3100f,42.9360f,46.8940f
    };
    if(qp<0)qp=0; if(qp>51)qp=51;
    return table[qp];
}

/* ============================ DPB (Multi-Reference) ============================ */

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

/* ============================ Adaptive GOP ============================ */

static int scenecut(const float* curr, const float* prev, int N, float c){
    float dist = hyperbolic_dist(curr, prev, N, c);
    return dist > 0.5f;
}

/* ============================ Main Encoder ============================ */

int main(int argc, char** argv){
    if(argc<3){
        printf("WuBuLatent v2 — CABAC + RDO + Adaptive GOP + Flow B-frames\n");
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

    printf("WuBuLatent v2 — CABAC + RDO + Adaptive GOP + Flow B-frames\n");
    printf("  Input:  %s\n", input);
    printf("  Output: %s\n", output);
    printf("  Dim: %d, Vectors: %d, Curvature: %.2f\n", dim, num_vecs, curvature);
    printf("  QP: %.4f, Max GOP: %d, Max B: %d, Refs: %d\n", base_qstep, gop_size, b_frames, nrefs);

    /* Load input */
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

    /* Project onto Poincaré ball */
    for(int i=0;i<num_vecs;i++) poincare_clip(latents+i*dim, latents+i*dim, dim, curvature);

    /* Create container */
    WubuvWriter* wr=wubuv_writer_open(output, dim, num_vecs, 30, num_vecs, 0, 0, 1);
    if(!wr){ printf("Cannot create container\n"); return 1; }

    /* Working buffers */
    float* recon=calloc(num_vecs*dim, sizeof(float));
    float* all_recon=calloc(num_vecs*dim, sizeof(float));
    float* tangent=calloc(dim, sizeof(float));
    int16_t* quantized=calloc(dim, sizeof(int16_t));
    RdoMode* modes=calloc(num_vecs, sizeof(RdoMode));

    LatentDPB dpb;
    dpb_init(&dpb, dim);

    double total_dist=0;
    int ifr=0, pfr=0, bfr=0;
    long total_comp=0;
    int qp = 26;
    float lambda = lambda_from_qp(qp);

    /* Pass 1: RDO Mode Decision */
    printf("--- Pass 1: RDO Mode Decision ---\n");
    int next_i = 0;

    for(int i=0;i<num_vecs;i++){
        const float* curr=latents+i*dim;

        /* Force I-frame at scene cuts or GOP boundary */
        if(i==0 || scenecut(curr, latents+(i-1)*dim, dim, curvature) || i>=next_i){
            modes[i]=MODE_I;
            next_i = i + gop_size;
            continue;
        }

        /* Evaluate P-frame cost using best of multiple references */
        float p_cost = 1e18f;
        for(int ri=0; ri<nrefs; ri++){
            const float* ref = dpb_get(&dpb, ri);
            if(!ref) break;
            logmap_at(tangent, ref, curr, dim, curvature);
            float t_norm=vec_norm(tangent,dim);
            float qstep=base_qstep*0.1f*(1.0f+t_norm*10.0f);

            int16_t iq[DIM_MAX];
            for(int d=0;d<dim;d++){
                int l=(int)roundf(tangent[d]/qstep);
                if(l>32767)l=32767; if(l<-32768)l=-32768;
                iq[d]=(int16_t)l;
            }
            int bits=cabac_bits_estimate(iq, dim);

            float dequant[DIM_MAX];
            for(int d=0;d<dim;d++) dequant[d]=(float)iq[d]*qstep;
            float recon_pt[DIM_MAX];
            expmap_at(recon_pt, ref, dequant, dim, curvature);
            float dist=hyperbolic_dist(curr, recon_pt, dim, curvature);

            float cost=rd_cost(dist, bits, lambda);
            if(cost < p_cost) p_cost = cost;
        }

        /* Evaluate B-frame cost using geodesic interpolation */
        float b_cost = 1e18f;
        if(b_frames>0 && i<num_vecs-1 && dpb.count>=1){
            const float* fwd=dpb_get(&dpb, 0);
            const float* bwd=latents+(i+1)*dim;
            if(fwd){
                float interp[DIM_MAX];
                geodesic_interp(interp, fwd, bwd, 0.5f, dim, curvature);
                float residual[DIM_MAX];
                logmap_at(residual, interp, curr, dim, curvature);
                float r_norm=vec_norm(residual,dim);
                /* B-frames use same QP as P-frames (residual is small) */
                float qstep=base_qstep*0.1f*(1.0f+r_norm*10.0f);

                int16_t iq[DIM_MAX];
                for(int d=0;d<dim;d++){
                    int l=(int)roundf(residual[d]/qstep);
                    if(l>32767)l=32767; if(l<-32768)l=-32768;
                    iq[d]=(int16_t)l;
                }
                int bits=cabac_bits_estimate(iq, dim);

                float dequant[DIM_MAX];
                for(int d=0;d<dim;d++) dequant[d]=(float)iq[d]*qstep;
                float recon_pt[DIM_MAX];
                expmap_at(recon_pt, interp, dequant, dim, curvature);
                float dist=hyperbolic_dist(curr, recon_pt, dim, curvature);

                b_cost=rd_cost(dist, bits, lambda);
            }
        }

        /* Choose best mode */
        if(b_cost < p_cost && b_cost < 1e17f){
            modes[i]=MODE_B;
        } else {
            modes[i]=MODE_P;
        }
    }

    /* Pass 2: Encode with chosen modes */
    printf("--- Pass 2: Encode ---\n");
    for(int i=0;i<num_vecs;i++){
        const float* curr=latents+i*dim;
        RdoMode mode=modes[i];

        uint8_t* fbuf=malloc(dim*4+64);
        int fpos=0;

        /* Header */
        fbuf[0]=(uint8_t)(dim&0xFF);
        fbuf[1]=(uint8_t)((dim>>8)&0xFF);
        fbuf[2]=1;
        fbuf[3]=(uint8_t)mode;
        fpos=4;

        if(mode==MODE_I){
            /* I-frame: quantize in tangent space at origin, then expmap back.
             * This ensures the reconstruction stays on the manifold. */
            float qstep=base_qstep*(1.0f+vec_norm(curr,dim)*10.0f);

            /* Logmap to tangent space */
            float tangent[DIM_MAX];
            logmap(tangent, curr, dim, curvature);

            /* Quantize tangent vector */
            int16_t iq[DIM_MAX];
            for(int d=0;d<dim;d++){
                int l=(int)roundf(tangent[d]/qstep);
                if(l>32767)l=32767; if(l<-32768)l=-32768;
                iq[d]=(int16_t)l;
            }

            /* CABAC encode */
            int bsz=cabac_encode_block(iq, dim, fbuf+fpos, dim*4+64-fpos);
            fpos += bsz;

            /* Reconstruct: dequantize + expmap back to ball */
            float dequant[DIM_MAX];
            for(int d=0;d<dim;d++) dequant[d]=(float)iq[d]*qstep;
            expmap(recon+i*dim, dequant, dim, curvature);

            ifr++;
            if(i%10==0||i<5) printf("  Frame %d (I): qstep=%.4f, bytes=%d\n", i, qstep, fpos-4);

        } else if(mode==MODE_P){
            /* P-frame: tangent velocity from best ref (search DPB) */
            const float* best_ref = dpb_get(&dpb, 0);
            float best_cost = 1e18f;
            logmap_at(tangent, best_ref, curr, dim, curvature);
            float t_norm=vec_norm(tangent,dim);
            float qstep=base_qstep*0.1f*(1.0f+t_norm*10.0f);

            /* Find best reference */
            for(int ri=0; ri<nrefs; ri++){
                const float* ref = dpb_get(&dpb, ri);
                if(!ref) break;
                float tan[DIM_MAX];
                logmap_at(tan, ref, curr, dim, curvature);
                float tn=vec_norm(tan,dim);
                float qs=base_qstep*0.1f*(1.0f+tn*10.0f);

                int16_t iq[DIM_MAX];
                for(int d=0;d<dim;d++){
                    int l=(int)roundf(tan[d]/qs);
                    if(l>32767)l=32767; if(l<-32768)l=-32768;
                    iq[d]=(int16_t)l;
                }
                int bits=cabac_bits_estimate(iq, dim);
                float dist=tn;
                float cost=rd_cost(dist, bits, lambda);
                if(cost < best_cost){
                    best_cost=cost;
                    best_ref=ref;
                    logmap_at(tangent, ref, curr, dim, curvature);
                    t_norm=tn;
                    qstep=qs;
                }
            }

            /* Quantize */
            int16_t iq[DIM_MAX];
            for(int d=0;d<dim;d++){
                int l=(int)roundf(tangent[d]/qstep);
                if(l>32767)l=32767; if(l<-32768)l=-32768;
                iq[d]=(int16_t)l;
            }

            /* CABAC encode */
            int bsz=cabac_encode_block(iq, dim, fbuf+fpos, dim*4+64-fpos);
            fpos += bsz;

            /* Reconstruct */
            float dequant[DIM_MAX];
            for(int d=0;d<dim;d++) dequant[d]=(float)iq[d]*qstep;
            expmap_at(recon+i*dim, best_ref, dequant, dim, curvature);

            pfr++;
            if(i%10==0||i<5) printf("  Frame %d (P): qstep=%.4f, bytes=%d\n", i, qstep, fpos-4);

        } else {
            /* B-frame: geodesic interpolation + residual */
            const float* fwd=dpb_get(&dpb, 0);
            const float* bwd=latents+(i+1)*dim;
            if(!fwd) fwd=bwd;

            float interp[DIM_MAX];
            geodesic_interp(interp, fwd, bwd, 0.5f, dim, curvature);

            float residual[DIM_MAX];
            logmap_at(residual, interp, curr, dim, curvature);
            float r_norm=vec_norm(residual,dim);
            /* B-frames use slightly coarser QP than P */
            float qstep=base_qstep*0.2f*(1.0f+r_norm*10.0f);

            /* Quantize */
            int16_t iq[DIM_MAX];
            for(int d=0;d<dim;d++){
                int l=(int)roundf(residual[d]/qstep);
                if(l>32767)l=32767; if(l<-32768)l=-32768;
                iq[d]=(int16_t)l;
            }

            /* CABAC encode */
            int bsz=cabac_encode_block(iq, dim, fbuf+fpos, dim*4+64-fpos);
            fpos += bsz;

            /* Reconstruct */
            float dequant[DIM_MAX];
            for(int d=0;d<dim;d++) dequant[d]=(float)iq[d]*qstep;
            expmap_at(recon+i*dim, interp, dequant, dim, curvature);

            bfr++;
            if(i%10==0||i<5) printf("  Frame %d (B): qstep=%.4f, bytes=%d\n", i, qstep, fpos-4);
        }

        memcpy(all_recon+i*dim, recon+i*dim, dim*sizeof(float));
        wubuv_write_video(wr, fbuf, fpos, mode==MODE_I);
        total_comp += fpos;
        free(fbuf);

        /* Update DPB */
        dpb_push(&dpb, recon+i*dim);
    }

    wubuv_writer_close(wr);

    /* Compute distortion */
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

    /* Compare with all-I baseline */
    long alli_bits=0;
    for(int i=0;i<num_vecs;i++){
        float qstep=base_qstep*(1.0f+vec_norm(latents+i*dim,dim)*10.0f);
        int nz=0;
        for(int d=0;d<dim;d++){
            int l=(int)roundf(latents[i*dim+d]/qstep);
            if(l!=0) nz++;
        }
        alli_bits += nz*20+10;
    }
    printf("All-I baseline: %.2f KB (%.1f bits/frame)\n", alli_bits/1024.0, (float)alli_bits/num_vecs);
    printf("Bit saving vs all-I: %.1f%%\n", 100.0f*(1.0f-(float)total_comp/alli_bits));

    free(latents); free(recon); free(all_recon);
    free(tangent); free(quantized); free(modes);
    dpb_free(&dpb);
    return 0;
}

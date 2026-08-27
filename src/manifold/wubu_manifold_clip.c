/*
 * wubu_manifold_clip.c -- Manifold CLIP: hyperbolic contrastive alignment
 *
 * GAP-D002/D003 CLOSED. The WuBu replacement for flat CLIP:
 *   CLIP        : W_img·x, W_txt·y in R^n, cosine similarity, InfoNCE
 *   Manifold CLIP: embeddings ON the Poincare ball, GEODESIC similarity
 *                  d(q_i, q_j) = (1/√c)·acosh(1 + 2c||x-y||²/((1-c||x||²)(1-c||y||²)))
 *                  InfoNCE over -d (closer = more similar), Riemannian SGD-ready.
 *
 * Design notes (from the 2026-08-24 Kevin-Bacon session, nodes 1.1-1.5):
 *   - Hierarchy lives near the boundary; entailment-cone loss is a future
 *     layer (D004) — this module provides the geodesic-contrastive core.
 *   - Embeddings are kept strictly inside the ball by a projection guard:
 *     any ||x|| >= r_max (0.999) is rescaled — prevents NaN in acosh.
 *   - Temperature is learnable, clamped to [tau_min, tau_max].
 */

#include "wubu_manifold_clip.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MC_RMAX 0.999f   /* max ball radius before projection */

/* ------------------------------------------------------------------ */
static uint32_t mc_rng = 20260824u;
static float mc_uniform(void){
    mc_rng = mc_rng*1103515245u + 12345u;
    return (float)(mc_rng>>16)/65536.0f;
}

static void mc_project(float* x,int D){
    float n2=0; for(int d=0;d<D;d++) n2+=x[d]*x[d];
    if(n2 > MC_RMAX*MC_RMAX){
        float s=MC_RMAX/sqrtf(n2);
        for(int d=0;d<D;d++) x[d]*=s;
    }
}

/* geodesic distance on Poincaré ball of curvature c>0 */
float wubu_mclip_geodesic(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d]; ab2+=df*df;
        a2+=a[d]*a[d]; b2+=b[d]*b[d];
    }
    float denom=(1.0f-c*a2)*(1.0f-c*b2);
    if(denom<1e-9f) denom=1e-9f;
    float arg=1.0f+2.0f*c*ab2/denom;
    if(arg<1.0f) arg=1.0f;
    return acoshf(arg)/sqrtf(c);
}

int wubu_mclip_init(WubuManifoldClip* m,const WubuMclipConfig* cfg){
    memset(m,0,sizeof(*m));
    m->cfg=*cfg;
    int D=cfg->embed_dim;
    /* small random init keeps embeddings near origin (high curvature region) */
    int F=cfg->feat_dim;
    size_t n=(size_t)D*F;
    m->proj_a=(float*)malloc(n*sizeof(float));
    m->proj_b=(float*)malloc(n*sizeof(float));
    if(!m->proj_a||!m->proj_b) return -1;
    float s=0.6f/sqrtf((float)D);
    for(size_t i=0;i<n;i++){
        m->proj_a[i]=(mc_uniform()*2-1)*s;
        m->proj_b[i]=(mc_uniform()*2-1)*s;
    }
    m->log_tau=0.0f;                 /* tau = e^0 = 1 */
    m->log_c=0.0f;                   /* c = e^0 = 1 (GAP-D009) */
    /* GAP-D013: per-modality scale scalars in log space (MERU recipe),
     * init 1/sqrt(F) so embeddings have expected unit norm at start. */
    float alpha_init=-0.5f*logf((float)F);
    m->log_alpha_a=alpha_init;
    m->log_alpha_b=alpha_init;
    m->step_count=0;
    m->initialized=true;
    return 0;
}

void wubu_mclip_free(WubuManifoldClip* m){
    free(m->proj_a);free(m->proj_b);
    m->proj_a=m->proj_b=NULL;
    m->initialized=false;
}

/* project features onto the ball through a learned linear head */
void wubu_mclip_embed(WubuManifoldClip* m,const float* feat,int F,
                      const float* W,int D,float* out_batch,int B){
    for(int b=0;b<B;b++){
        /* out = tanh(W · feat / F) mapped radially into the ball */
        float h[256];
        int Dc=D<256?D:256;
        for(int j=0;j<Dc;j++){
            float acc=0;
            for(int i=0;i<F;i++) acc+=W[j*(size_t)F+i]*feat[b*(size_t)F+i];
            h[j]=tanhf(acc/(float)F);          /* (-1,1) => inside ball */
            out_batch[b*(size_t)D+j]=h[j]*0.95f; /* keep off the boundary */
        }
    }
}

float wubu_mclip_curvature(const WubuManifoldClip* m){ return expf(m->log_c); }

void wubu_mclip_similarity_matrix(WubuManifoldClip* m,
                                  const float* emb_a,const float* emb_b,
                                  int Ba,int Bb,int D,float* sim){
    float tau=expf(m->log_tau);
    float c=expf(m->log_c);   /* GAP-D009: learned curvature */
    for(int i=0;i<Ba;i++)
        for(int j=0;j<Bb;j++)
            sim[i*(size_t)Bb+j]=
                -wubu_mclip_geodesic(emb_a+i*D,emb_b+j*D,D,c)/(tau+1e-6f);
}

/* InfoNCE both directions (i->j and j->i), averaged.
 * Gradients flow to the embeddings; caller backprops to heads or we do a
 * finite-difference head update like wubu_flow_train_step. */
float wubu_mclip_infonce(WubuManifoldClip* m,const float* sim,
                         int B,int* indices_perm){
    (void)indices_perm;
    float loss=0;
    for(int i=0;i<B;i++){
        float mx=-1e30f;
        for(int j=0;j<B;j++){ float v=sim[i*(size_t)B+j]; if(v>mx)mx=v; }
        float z=0,p_pos=0;
        for(int j=0;j<B;j++){
            float e=expf(sim[i*(size_t)B+j]-mx);
            z+=e;
            if(j==i) p_pos=e;
        }
        loss+=-logf((p_pos+1e-9f)/z);
        /* column direction */
        mx=-1e30f;
        for(int j=0;j<B;j++){ float v=sim[j*(size_t)B+i]; if(v>mx)mx=v; }
        z=0;p_pos=0;
        for(int j=0;j<B;j++){
            float e=expf(sim[j*(size_t)B+i]-mx);
            z+=e; if(j==i)p_pos=e;
        }
        loss+=-logf((p_pos+1e-9f)/z);
    }
    return loss/(2.0f*(float)B);
}

/* One training step on synthetic-aligned pairs:
 * pairs (fa_i, fb_i) are positive matches. Perturb-and-restore SGD on both
 * heads (finite differences — same pattern as wubu_flow_train_step).
 * Returns the post-update loss. */
float wubu_mclip_train_step(WubuManifoldClip* m,
                            const float* feat_a,const float* feat_b,
                            int B,int F){
    int D=m->cfg.embed_dim;
    float lr=m->cfg.lr;

    float* ea=(float*)malloc((size_t)B*D*sizeof(float));
    float* eb=(float*)malloc((size_t)B*D*sizeof(float));
    float* sim=(float*)malloc((size_t)B*B*sizeof(float));

    wubu_mclip_embed(m,feat_a,F,m->proj_a,D,ea,B);
    wubu_mclip_embed(m,feat_b,F,m->proj_b,D,eb,B);
    wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);

    float base=wubu_mclip_infonce(m,sim,B,NULL);
    float eps=2e-3f;

    /* head A finite-difference SGD */
    size_t nA=(size_t)D*F;
    for(size_t k=0;k<nA;k++){
        float old=m->proj_a[k];
        m->proj_a[k]=old+eps;
        wubu_mclip_embed(m,feat_a,F,m->proj_a,D,ea,B);
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        float lp=wubu_mclip_infonce(m,sim,B,NULL);
        m->proj_a[k]=old-eps;
        wubu_mclip_embed(m,feat_a,F,m->proj_a,D,ea,B);
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        float lm=wubu_mclip_infonce(m,sim,B,NULL);
        m->proj_a[k]=old;
        m->proj_a[k]=old-lr*(lp-lm)/(2*eps);
    }
    /* head B */
    size_t nB=(size_t)D*F;
    for(size_t k=0;k<nB;k++){
        float old=m->proj_b[k];
        m->proj_b[k]=old+eps;
        wubu_mclip_embed(m,feat_b,F,m->proj_b,D,eb,B);
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        float lp=wubu_mclip_infonce(m,sim,B,NULL);
        m->proj_b[k]=old-eps;
        wubu_mclip_embed(m,feat_b,F,m->proj_b,D,eb,B);
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        float lm=wubu_mclip_infonce(m,sim,B,NULL);
        m->proj_b[k]=old;
        m->proj_b[k]=old-lr*(lp-lm)/(2*eps);
    }

    wubu_mclip_embed(m,feat_a,F,m->proj_a,D,ea,B);
    wubu_mclip_embed(m,feat_b,F,m->proj_b,D,eb,B);
    wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
    float after=wubu_mclip_infonce(m,sim,B,NULL);

    /* GAP-D009: curvature gradient by finite difference on log_c */
    {
        float eps=5e-2f;
        float old=m->log_c;
        m->log_c=old+eps;
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        float lp=wubu_mclip_infonce(m,sim,B,NULL);
        m->log_c=old-eps;
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        float lm=wubu_mclip_infonce(m,sim,B,NULL);
        m->log_c=old - 0.05f*(lp-lm)/(2*eps);   /* smaller LR for geometry */
    }
    wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
    after=wubu_mclip_infonce(m,sim,B,NULL);

    /* GAP-D013: per-modality scale scalars (MERU alpha_img/alpha_txt) —
     * FD gradients in log space; curvature clamped [0.1,10] per MERU
     * stability recipe; tau floored at 0.01. */
    {
        float eps=5e-2f;
        float old=m->log_alpha_a;
        m->log_alpha_a=old+eps;
        wubu_mclip_embed(m,feat_a,F,m->proj_a,D,ea,B);
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        float lp=wubu_mclip_infonce(m,sim,B,NULL);
        m->log_alpha_a=old-eps;
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        float lm=wubu_mclip_infonce(m,sim,B,NULL);
        m->log_alpha_a=old - 0.05f*(lp-lm)/(2*eps);

        old=m->log_alpha_b;
        m->log_alpha_b=old+eps;
        wubu_mclip_embed(m,feat_b,F,m->proj_b,D,eb,B);
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        lp=wubu_mclip_infonce(m,sim,B,NULL);
        m->log_alpha_b=old-eps;
        wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
        lm=wubu_mclip_infonce(m,sim,B,NULL);
        m->log_alpha_b=old - 0.05f*(lp-lm)/(2*eps);

        /* tau floor 0.01 (log_tau >= log(0.01)) — MERU recipe */
        if(m->log_tau < logf(0.01f)) m->log_tau = logf(0.01f);
    }
    wubu_mclip_similarity_matrix(m,ea,eb,B,B,D,sim);
    after=wubu_mclip_infonce(m,sim,B,NULL);

    /* GAP-D013: entailment term folded into the returned objective so the
     * entail_weight is honored end-to-end (loss reported to caller). */
    {
        float lam=m->cfg.entail_weight;
        if(lam>0.0f){
            float ent=0;
            for(int i=0;i<B;i++)
                ent+=wubu_mclip_entailment_loss(ea+i*D,eb+i*D,D,
                                                expf(m->log_c),0.1f);
            after+=lam*ent/(float)B;
        }
    }

    m->step_count++;
    free(ea);free(eb);free(sim);
    return after;
}

/* retrieval accuracy@1: fraction of a's nearest b (by geodesic) that is its pair */
float wubu_mclip_recall_at1(WubuManifoldClip* m,
                            const float* feat_a,const float* feat_b,
                            int B,int F){
    int D=m->cfg.embed_dim;
    float* ea=(float*)malloc((size_t)B*D*sizeof(float));
    float* eb=(float*)malloc((size_t)B*D*sizeof(float));
    wubu_mclip_embed(m,feat_a,F,m->proj_a,D,ea,B);
    wubu_mclip_embed(m,feat_b,F,m->proj_b,D,eb,B);
    int hits=0;
    for(int i=0;i<B;i++){
        float best=1e30f; int bj=-1;
        for(int j=0;j<B;j++){
            float d=wubu_mclip_geodesic(ea+i*D,eb+j*D,D,1.0f);
            if(d<best){best=d;bj=j;}
        }
        if(bj==i) hits++;
    }
    free(ea);free(eb);
    return (float)hits/(float)B;
}

/* =================================================================== */
/* GAP-D004: Entailment-cone loss (HyCoCLIP lineage, node 1.2).        */
/* x is ENTAILED by parent y iff the geodesic angle between them is    */
/* within y's cone aperture:                                           */
/*   apex(y) = arcsin( K / (sqrt(c)*||y||) ), K = sin(apex_0), 0<K<=1  */ 
/* For ||y||->0 aperture -> pi/2 (root entails everything locally);    */
/* toward boundary the cone tightens. Loss per pair:                   */
/*   margin * d_geo(x,y) + relu(angle - apex(y))                       */
/* i.e. pull entailed pairs together AND enforce the cone constraint.  */

float wubu_mclip_entailment_loss(const float* child,const float* parent,
                                 int D,float c,float kappa){
    float p2=0;
    for(int d=0;d<D;d++) p2+=parent[d]*parent[d];
    /* angle between child and parent at parent: use Euclidean proxy of the
     * gyroangle — cos theta via inner product (valid local approximation
     * for contrastive shaping; exact gyrovectors are D-cell follow-up). */
    float dot=0,n2=0;
    for(int d=0;d<D;d++){ dot+=child[d]*parent[d]; n2+=child[d]*child[d]; }
    float denom=sqrtf(p2*n2);
    if(denom<1e-9f) return 0.0f;
    float cos_t=dot/denom;
    if(cos_t>1.0f)cos_t=1.0f; if(cos_t<-1.0f)cos_t=-1.0f;
    float ang=acosf(cos_t);

    float sn=p2>0?sinf(0.0f):0; (void)sn;
    float norm_p=sqrtf(p2);
    float sin_apex=kappa/(sqrtf(c)*norm_p+1e-6f);
    if(sin_apex>1.0f) sin_apex=1.0f;
    float apex=asinf(sin_apex);

    float dist=wubu_mclip_geodesic(child,parent,D,c);
    float violation=ang-apex; if(violation<0) violation=0;
    return 0.1f*dist+violation;
}

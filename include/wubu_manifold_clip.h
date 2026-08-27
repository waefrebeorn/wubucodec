/*
 * wubu_manifold_clip.h -- Manifold CLIP: hyperbolic contrastive alignment
 *
 * GAP-D002 (InfoNCE) + GAP-D003 (geodesic similarity) CLOSED.
 * The WuBu replacement for CLIP's flat linear algebra:
 *   embeddings live ON the Poincaré ball; similarity = negative geodesic
 *   distance; contrastive loss = symmetric InfoNCE; heads train by
 *   finite-difference SGD (pattern-consistent with wubu_flow_train_step).
 */

#ifndef WUBU_MANIFOLD_CLIP_H
#define WUBU_MANIFOLD_CLIP_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int embed_dim;   /* ball dimension D */
    int feat_dim;    /* input feature length F */
    float lr;        /* head learning rate */
    float entail_weight;  /* GAP-D013: lambda on entailment loss (MERU: 0.2) */
} WubuMclipConfig;

typedef struct {
    WubuMclipConfig cfg;
    float* proj_a;      /* [D,F] modality-A head (image)  */
    float* proj_b;      /* [D,F] modality-B head (text/audio) */
    float log_tau;      /* learnable temperature, tau = exp(log_tau), floor 0.01 */
    float log_c;        /* GAP-D009: learnable curvature, c = exp(log_c),
                           clamped [0.1,10] per MERU stability recipe */
    float log_alpha_a;  /* GAP-D013: per-modality scale scalars (log space,
                           MERU alpha_img/alpha_txt, init 1/sqrt(F)) */
    float log_alpha_b;
    int step_count;
    bool initialized;
} WubuManifoldClip;

int  wubu_mclip_init(WubuManifoldClip* m, const WubuMclipConfig* cfg);
void wubu_mclip_free(WubuManifoldClip* m);

/* Learned curvature accessor (GAP-D009): c = exp(log_c). */
float wubu_mclip_curvature(const WubuManifoldClip* m);

/* Geodesic distance on the Poincaré ball (curvature c). GAP-D003. */
float wubu_mclip_geodesic(const float* a, const float* b, int D, float c);

/* Project B feature rows [B,F] through head W into ball embeddings [B,D]. */
void wubu_mclip_embed(WubuManifoldClip* m, const float* feat, int F,
                      const float* W, int D, float* out_batch, int B);

/* Similarity matrix sim[i,j] = -geodesic(a_i,b_j)/tau. */
void wubu_mclip_similarity_matrix(WubuManifoldClip* m,
                                  const float* emb_a, const float* emb_b,
                                  int Ba, int Bb, int D, float* sim);

/* Symmetric InfoNCE over a B×B similarity matrix. GAP-D002. */
float wubu_mclip_infonce(WubuManifoldClip* m, const float* sim,
                         int B, int* indices_perm);

/* One training step on aligned pairs; returns post-update loss. */
float wubu_mclip_train_step(WubuManifoldClip* m,
                            const float* feat_a, const float* feat_b,
                            int B, int F);

/* GAP-D004: entailment-cone loss — child inside parent's cone.
 * kappa in (0,1]: cone openness at boundary. */
float wubu_mclip_entailment_loss(const float* child, const float* parent,
                                 int D, float c, float kappa);

/* Retrieval accuracy@1 under geodesic nearest neighbor. */
float wubu_mclip_recall_at1(WubuManifoldClip* m,
                            const float* feat_a, const float* feat_b,
                            int B, int F);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MANIFOLD_CLIP_H */

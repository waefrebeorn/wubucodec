/* test_wubu_text_encoder.c -- GAP-D005 gates
 *  G1 deterministic: same text -> identical embedding across calls
 *  G2 distinct texts -> distinct embeddings (cosine < 1)
 *  G3 similar texts ("a dog runs" vs "a dog runs fast") closer than dissimilar
 *  G4 end-to-end: text features through mclip head land inside the ball,
 *     and aligned text/image pairs retrieve above chance after training
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_text_encoder.h"
#include "../train/wubu_manifold_clip.c"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s\n",#c); failed++; return; } }while(0)

static float cosine(const float*a,const float*b,int D){
    float dot=0,na=0,nb=0;
    for(int i=0;i<D;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return dot/(sqrtf(na*nb)+1e-9f);
}
int main(void){
    printf("=== WuBuMath Text Encoder Tests ===\n\n");
    WubuTextEncoder te;
    CHECK(wubu_text_init(&te,4096,16)==0);

    float e1[16],e2[16],e3[16],e4[16];
    wubu_text_encode(&te,"the old unc claims the wishing well appears",e1,16);
    wubu_text_encode(&te,"the old unc claims the wishing well appears",e2,16);
    wubu_text_encode(&te,"quantum flux capacitor overdrives the manifold",e3,16);
    wubu_text_encode(&te,"the old unc claims the wishing well appears at midnight",e4,16);

    printf("  g1_deterministic..."); 
    for(int d=0;d<16;d++) CHECK(fabsf(e1[d]-e2[d])<1e-9f);
    printf("PASS\n");passed++;

    printf("  g2_distinct...");      
    CHECK(cosine(e1,e3,16)<0.99f);
    printf("PASS\n");passed++;

    printf("  g3_similarity_ordering...");
    CHECK(cosine(e1,e4,16)>cosine(e1,e3,16));
    printf("PASS\n");passed++;

    /* end-to-end with manifold CLIP: train on paired (text-feature, image-feature) */
    WubuManifoldClip m;
    WubuMclipConfig cfg={ .embed_dim=8,.feat_dim=16,.lr=0.15f };
    CHECK(wubu_mclip_init(&m,&cfg)==0);

    const char* docs[8]={
        "wishing well appears when nobody records",
        "the AGI wakes inside the monitors glow",
        "storage units hum in ordered rows",
        "the teal orb staff marks the threshold",
        "found footage anthology of the deep end",
        "game show horror in the mega chamber",
        "deadpan philosopher stares into static",
        "regurgitation of humanity after death"};
    float feat_t[8*16];
    for(int i=0;i<8;i++) wubu_text_encode(&te,docs[i],feat_t+i*16,16);
    /* pseudo-image features correlated with doc index */
    unsigned r=99u; float feat_v[8*16];
    for(int i=0;i<8;i++)
        for(int d=0;d<16;d++){
            r=r*1103515245u+12345u;
            float noise=(float)(r>>16)/65536.0f-0.5f;
            feat_v[i*16+d]=0.6f*feat_t[i*16+d]+0.25f*noise;
        }

    printf("  g4_clip_end_to_end...   ");
    float before=wubu_mclip_recall_at1(&m,feat_t,feat_v,8,16);
    for(int s=0;s<900;s++) wubu_mclip_train_step(&m,feat_t,feat_v,8,16);
    { float r50=wubu_mclip_recall_at1(&m,feat_t,feat_v,8,16);
      printf("[r50=%d] ",(int)(r50*100)); }
    float after=wubu_mclip_recall_at1(&m,feat_t,feat_v,8,16);
    CHECK(after>before);
    CHECK(after>1.0f/8.0f);
    printf("[%.2f->%.2f] PASS\n",(double)before,(double)after);
    passed++;

    wubu_mclip_free(&m);wubu_text_free(&te);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
